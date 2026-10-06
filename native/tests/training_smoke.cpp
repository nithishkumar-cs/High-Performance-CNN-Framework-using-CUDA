#include "detector/trainer.hpp"
#include "detector/yolo_loss.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        auto directory = std::filesystem::path(__FILE__).parent_path();
        auto network = detector::Network::from_cfg((directory / "train-mini.cfg").string());
        network.initialize_random(7);
        detector::Trainer trainer(network, detector::LearningRateSchedule{0.1f}, 1, 0.0f);
        std::vector<float> input{1.0f, 0.5f, 0.25f};
        std::vector<detector::Target> targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};
        float first = trainer.step({input}, {targets}).loss;
        float last = first;
        for (int i = 0; i < 99; ++i) last = trainer.step({input}, {targets}).loss;
        std::cout << "first_loss=" << first << " last_loss=" << last << "\n";
        expect(std::isfinite(last) && last < first - 0.5f, "single-box loss did not fall");
        auto expected = network.forward(input, detector::Backend::cpu)[0].values;
        auto checkpoint = std::filesystem::current_path() / "detector_training_smoke.weights";
        network.save_weights(checkpoint.string());
        auto restored = detector::Network::from_cfg((directory / "train-mini.cfg").string());
        restored.load_weights(checkpoint.string());
        std::filesystem::remove(checkpoint);
        auto actual = restored.forward(input, detector::Backend::cpu)[0].values;
        for (std::size_t i = 0; i < actual.size(); ++i)
            expect(std::abs(actual[i] - expected[i]) < 1e-6f, "checkpoint round trip changed predictions");

        auto graph = detector::Network::from_cfg((directory / "mini.cfg").string());
        graph.initialize_random(9);
        detector::Trainer graph_trainer(graph, detector::LearningRateSchedule{0.0001f}, 1);
        auto result = graph_trainer.step({std::vector<float>(graph.input_shape().elements(), 0.2f)},
                                         {{{0, 0.5f, 0.5f, 0.25f, 0.25f}}});
        expect(std::isfinite(result.loss) && result.positives == 1,
               "branched graph (batch-normalize + route + upsample) training step");

        // Shortcut (residual) layer: forward+backward must run and reduce loss on a repeated target,
        // exercising fan-out gradient accumulation into both the shortcut's previous layer and branch.
        auto shortcut_net = detector::Network::from_cfg((directory / "shortcut-train.cfg").string());
        shortcut_net.initialize_random(11);
        detector::Trainer shortcut_trainer(shortcut_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        std::vector<float> shortcut_input{0.3f};
        std::vector<detector::Target> shortcut_targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};
        float shortcut_first = shortcut_trainer.step({shortcut_input}, {shortcut_targets}).loss;
        float shortcut_last = shortcut_first;
        for (int i = 0; i < 49; ++i) shortcut_last = shortcut_trainer.step({shortcut_input}, {shortcut_targets}).loss;
        expect(std::isfinite(shortcut_last) && shortcut_last < shortcut_first, "shortcut layer loss did not fall");

        // True minibatch batch-norm: a single step() call with `batch_size` identical copies of
        // one sample must learn exactly as a batch_size=1 trainer does on that one sample, since
        // batch statistics over N identical copies equal that one sample's own statistics, and
        // averaging N identical gradients equals the single gradient. Uses mini.cfg (has a
        // batch_normalize layer) so this actually exercises the batched statistics path.
        auto graph_input = std::vector<float>(graph.input_shape().elements(), 0.2f);
        std::vector<detector::Target> graph_targets{{0, 0.5f, 0.5f, 0.25f, 0.25f}};
        auto reference_net = detector::Network::from_cfg((directory / "mini.cfg").string());
        reference_net.initialize_random(21);
        detector::Trainer reference_trainer(reference_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        reference_trainer.step({graph_input}, {graph_targets});

        auto batch_net = detector::Network::from_cfg((directory / "mini.cfg").string());
        batch_net.initialize_random(21);
        detector::Trainer batch_trainer(batch_net, detector::LearningRateSchedule{0.05f}, 4, 0.0f);
        batch_trainer.step(std::vector<std::vector<float>>(4, graph_input), std::vector<std::vector<detector::Target>>(4, graph_targets));
        expect(batch_net.layers()[0].weights.size() == reference_net.layers()[0].weights.size(), "weight count mismatch");
        for (std::size_t p = 0; p < batch_net.layers()[0].weights.size(); ++p)
            expect(std::abs(batch_net.layers()[0].weights[p] - reference_net.layers()[0].weights[p]) < 1e-5f,
                   "batch of identical samples did not match single-sample training");

        // Learning-rate schedule: linear burn-in then a step decay, matching Darknet's policy.
        detector::LearningRateSchedule schedule{1.0f, 4, 4.0f, {10}, {0.1f}};
        expect(schedule.at(0) < schedule.at(1) && schedule.at(1) < schedule.at(3), "burn-in did not ramp up");
        expect(std::abs(schedule.at(3) - 1.0f) < 1e-6f, "burn-in should reach base_lr at its last iteration");
        expect(std::abs(schedule.at(4) - 1.0f) < 1e-6f, "post burn-in learning rate should equal base_lr");
        expect(std::abs(schedule.at(10) - 0.1f) < 1e-6f, "step decay did not scale learning rate");

        // Finite-difference gradient check for the new trainable batch-norm backward: perturb
        // gamma (scale) and one convolution weight, compare the numeric loss-slope against the
        // weight change Trainer's analytic backward actually applied (update = lr * grad).
        auto bn_cfg = (directory / "bn-grad-check.cfg").string();
        std::vector<float> gc_input{0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<detector::Target> gc_targets{{0, 0.5f, 0.5f, 0.3f, 0.3f}};
        auto loss_with = [&](int param, float delta) {
            auto net = detector::Network::from_cfg(bn_cfg);
            net.initialize_random(5);
            if (param == 0) net.layer(0).scales[0] += delta;
            else if (param == 1) net.layer(0).biases[0] += delta;
            else net.layer(0).weights[0] += delta;
            detector::Trainer t(net, detector::LearningRateSchedule{1e-9f}, 1, 0.0f);
            return t.step({gc_input}, {gc_targets}).loss;
        };
        auto analytic_grad = [&](int param) {
            auto net = detector::Network::from_cfg(bn_cfg);
            net.initialize_random(5);
            float before = param == 0 ? net.layer(0).scales[0] : param == 1 ? net.layer(0).biases[0] : net.layer(0).weights[0];
            float lr = 1e-4f;
            detector::Trainer t(net, detector::LearningRateSchedule{lr}, 1, 0.0f);
            t.step({gc_input}, {gc_targets});
            float after = param == 0 ? net.layer(0).scales[0] : param == 1 ? net.layer(0).biases[0] : net.layer(0).weights[0];
            return (before - after) / lr;
        };
        const float eps = 1e-3f;
        const char* names[] = {"gamma", "beta", "weight"};
        for (int param = 0; param < 3; ++param) {
            float numeric = (loss_with(param, eps) - loss_with(param, -eps)) / (2 * eps);
            float analytic = analytic_grad(param);
            float tolerance = 2e-2f * std::max(1.0f, std::abs(numeric));
            std::cout << "bn_grad_check " << names[param] << " numeric=" << numeric << " analytic=" << analytic << '\n';
            expect(std::abs(numeric - analytic) < tolerance, "batch-norm backward gradient check failed");
        }

        // Finite-difference gradient check for TRUE minibatch batch-norm statistics: two
        // *different* samples in one step() call, so batch mean/variance differ from either
        // sample's own statistics and the backward sums must correctly combine both samples'
        // contributions (the total_n = n*plane normalizer). step()'s returned loss is already
        // summed over the batch, so its finite difference is directly comparable to the SGD
        // update's accumulated (not averaged) gradient, i.e. (before - after) / lr * n.
        std::vector<float> gc_input_b{0.4f, 0.1f, 0.3f, 0.2f};
        std::vector<detector::Target> gc_targets_b{{0, 0.3f, 0.6f, 0.2f, 0.4f}};
        auto batch_loss_with = [&](int param, float delta) {
            auto net = detector::Network::from_cfg(bn_cfg);
            net.initialize_random(5);
            if (param == 0) net.layer(0).scales[0] += delta;
            else if (param == 1) net.layer(0).biases[0] += delta;
            else net.layer(0).weights[0] += delta;
            detector::Trainer t(net, detector::LearningRateSchedule{1e-9f}, 2, 0.0f);
            return t.step({gc_input, gc_input_b}, {gc_targets, gc_targets_b}).loss;
        };
        auto batch_analytic_grad = [&](int param) {
            auto net = detector::Network::from_cfg(bn_cfg);
            net.initialize_random(5);
            float before = param == 0 ? net.layer(0).scales[0] : param == 1 ? net.layer(0).biases[0] : net.layer(0).weights[0];
            float lr = 1e-4f;
            detector::Trainer t(net, detector::LearningRateSchedule{lr}, 2, 0.0f);
            t.step({gc_input, gc_input_b}, {gc_targets, gc_targets_b});
            float after = param == 0 ? net.layer(0).scales[0] : param == 1 ? net.layer(0).biases[0] : net.layer(0).weights[0];
            return (before - after) / lr * 2.0f;
        };
        for (int param = 0; param < 3; ++param) {
            float numeric = (batch_loss_with(param, eps) - batch_loss_with(param, -eps)) / (2 * eps);
            float analytic = batch_analytic_grad(param);
            float tolerance = 2e-2f * std::max(1.0f, std::abs(numeric));
            std::cout << "batch_bn_grad_check " << names[param] << " numeric=" << numeric << " analytic=" << analytic << '\n';
            expect(std::abs(numeric - analytic) < tolerance, "true minibatch batch-norm backward gradient check failed");
        }

        // ignore_thresh: a non-assigned anchor at the target's cell whose decoded predicted box
        // already matches the target exactly (IoU 1.0) must have its objectness-negative loss and
        // gradient suppressed when ignore_thresh (0.5 in this cfg) is met, and NOT suppressed when
        // ignore_thresh is raised past 1.0 (disabled), matching yolo_loss_and_gradient's contract.
        {
            auto build_output = [] {
                // attrs = classes(1) + 5 = 6, area = 4x4 = 16; anchor0 (mask slot 0, anchors[0]=(1,1))
                // at cell (x=1,y=1)=cell 5 is crafted to decode to cx=1.5,cy=1.5,w=3,h=3 (x/y logits
                // 0 => sigmoid 0.5 offset; w/h logits ln(3) => exp(ln3)*anchor(1,1)=3), exactly the
                // target box below, so its predicted box has IoU 1.0 with the target.
                std::vector<float> output(12 * 16, 0.0f);
                auto at = [&](int anchor, int attr, int cell) -> float& { return output[(static_cast<std::size_t>(anchor) * 6 + attr) * 16 + cell]; };
                at(0, 2, 5) = std::log(3.0f);
                at(0, 3, 5) = std::log(3.0f);
                at(0, 4, 5) = 2.0f;  // arbitrary nonzero objectness logit so a suppressed gradient is visibly different from an applied one
                return output;
            };
            std::vector<detector::Target> ignore_targets{{0, 0.375f, 0.375f, 0.75f, 0.75f}};
            auto ignore_cfg = (directory / "ignore-thresh.cfg").string();
            const std::size_t yolo_index = 1;  // [convolutional] is layer 0, [yolo] is layer 1

            auto net_suppressed = detector::Network::from_cfg(ignore_cfg);  // ignore_thresh=0.5 from cfg
            std::vector<std::vector<float>> output_suppressed(net_suppressed.layers().size());
            output_suppressed[yolo_index] = build_output();
            std::vector<std::vector<float>> grad_suppressed(net_suppressed.layers().size());
            grad_suppressed[yolo_index].assign(12 * 16, 0.0f);
            yolo_loss_and_gradient(net_suppressed, ignore_targets, output_suppressed, grad_suppressed);
            expect(grad_suppressed[yolo_index][(0 * 6 + 4) * 16 + 5] == 0.0f, "ignore_thresh did not suppress near-miss negative gradient");

            auto net_disabled = detector::Network::from_cfg(ignore_cfg);
            net_disabled.layer(yolo_index).ignore_thresh = 1.0f;  // disable: restores pre-ignore_thresh behavior
            std::vector<std::vector<float>> output_disabled(net_disabled.layers().size());
            output_disabled[yolo_index] = build_output();
            std::vector<std::vector<float>> grad_disabled(net_disabled.layers().size());
            grad_disabled[yolo_index].assign(12 * 16, 0.0f);
            yolo_loss_and_gradient(net_disabled, ignore_targets, output_disabled, grad_disabled);
            expect(grad_disabled[yolo_index][(0 * 6 + 4) * 16 + 5] != 0.0f, "disabled ignore_thresh unexpectedly suppressed the negative gradient");
        }

        std::cout << "training smoke passed: first=" << first << " last=" << last << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training smoke failed: " << error.what() << '\n';
        return 1;
    }
}


