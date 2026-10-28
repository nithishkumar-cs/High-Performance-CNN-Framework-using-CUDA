#include "detector/trainer.hpp"
#include "detector/l2norm.hpp"
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

/** Exercises detector training, analytic gradients, and YOLO loss invariants. */
int main() {
    try {
        std::vector<float> norm_input{0.3f, -0.7f, 1.2f, -0.4f};
        std::vector<float> norm_upstream{0.2f, -0.4f, 0.5f, 0.1f};
        std::vector<float> norm_output(norm_input.size()), norm_gradient(norm_input.size(), 0.0f);
        float inverse = detector::l2norm_forward(norm_input.data(), norm_output.data(), norm_input.size());
        detector::l2norm_backward(norm_output.data(), norm_upstream.data(), norm_gradient.data(),
                                  norm_input.size(), inverse);
        for (std::size_t i = 0; i < norm_input.size(); ++i) {
            constexpr float epsilon = 1e-3f;
            std::vector<float> plus = norm_input, minus = norm_input;
            plus[i] += epsilon;
            minus[i] -= epsilon;
            std::vector<float> plus_output(plus.size()), minus_output(minus.size());
            detector::l2norm_forward(plus.data(), plus_output.data(), plus.size());
            detector::l2norm_forward(minus.data(), minus_output.data(), minus.size());
            float plus_loss = 0.0f, minus_loss = 0.0f;
            for (std::size_t j = 0; j < norm_input.size(); ++j) {
                plus_loss += plus_output[j] * norm_upstream[j];
                minus_loss += minus_output[j] * norm_upstream[j];
            }
            float numeric = (plus_loss - minus_loss) / (2.0f * epsilon);
            expect(std::abs(norm_gradient[i] - numeric) < 2e-4f, "l2norm gradient check");
        }
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

        // Reorg (space-to-depth) layer: forward+backward must run and reduce loss through a
        // preceding conv's weights, exercising training.cpp's own copy of the reorg mapping
        // (independent from network.cpp's inference-only cpu_forward) and its backward (gradient
        // scatter back through the same bijection).
        auto reorg_net = detector::Network::from_cfg((directory / "reorg-train.cfg").string());
        reorg_net.initialize_random(13);
        detector::Trainer reorg_trainer(reorg_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        std::vector<float> reorg_input(reorg_net.input_shape().elements());
        for (std::size_t i = 0; i < reorg_input.size(); ++i) reorg_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        std::vector<detector::Target> reorg_targets{{0, 0.5f, 0.5f, 0.3f, 0.3f}};
        float reorg_first = reorg_trainer.step({reorg_input}, {reorg_targets}).loss;
        float reorg_last = reorg_first;
        for (int i = 0; i < 49; ++i) reorg_last = reorg_trainer.step({reorg_input}, {reorg_targets}).loss;
        expect(std::isfinite(reorg_last) && reorg_last < reorg_first, "reorg layer loss did not fall");

        // Dropout layer: forward+backward must run to a finite loss despite the random mask
        // (probability=0.1 in dropout-train.cfg keeps the step's gradient mostly intact, so loss
        // should still trend down over many steps even though each individual step is noisy).
        auto dropout_net = detector::Network::from_cfg((directory / "dropout-train.cfg").string());
        dropout_net.initialize_random(15);
        detector::Trainer dropout_trainer(dropout_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        std::vector<float> dropout_input{0.4f};
        std::vector<detector::Target> dropout_targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};
        float dropout_first = dropout_trainer.step({dropout_input}, {dropout_targets}).loss;
        float dropout_last = dropout_first;
        for (int i = 0; i < 199; ++i) {
            float loss = dropout_trainer.step({dropout_input}, {dropout_targets}).loss;
            expect(std::isfinite(loss), "dropout training step produced a non-finite loss");
            dropout_last = loss;
        }
        expect(dropout_last < dropout_first, "dropout layer loss did not trend down over 200 steps");

        // Region (YOLOv2-style) layer: forward+backward must run and reduce loss through a
        // preceding conv's weights, exercising the softmax class loss, rescore and scale-knob
        // branches in yolo_loss.cpp (shared with the CUDA trainer) that only [region] layers take.
        auto region_net = detector::Network::from_cfg((directory / "region-train.cfg").string());
        region_net.initialize_random(19);
        detector::Trainer region_trainer(region_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        std::vector<float> region_input(region_net.input_shape().elements());
        for (std::size_t i = 0; i < region_input.size(); ++i) region_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        std::vector<detector::Target> region_targets{{1, 0.5f, 0.5f, 0.3f, 0.3f}};
        float region_first = region_trainer.step({region_input}, {region_targets}).loss;
        float region_last = region_first;
        for (int i = 0; i < 49; ++i) region_last = region_trainer.step({region_input}, {region_targets}).loss;
        expect(std::isfinite(region_last) && region_last < region_first, "region layer loss did not fall");

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

        // Every unmatched box with a 0.5 objectness prediction needs a 0.5 gradient, even
        // when the head contains many candidates. Otherwise background learning vanishes as
        // the input resolution or number of YOLO heads grows.
        {
            auto background_net = detector::Network::from_cfg((directory / "ignore-thresh.cfg").string());
            std::vector<std::vector<float>> output(background_net.layers().size());
            std::vector<std::vector<float>> gradient(background_net.layers().size());
            output[1].assign(12 * 16, 0.0f);
            gradient[1].assign(12 * 16, 0.0f);
            yolo_loss_and_gradient(background_net, {}, output, gradient);
            expect(std::abs(gradient[1][(0 * 6 + 4) * 16] - 0.5f) < 1e-6f,
                   "background gradient was diluted by candidate count");
            expect(std::abs(gradient[1][(1 * 6 + 4) * 16 + 15] - 0.5f) < 1e-6f,
                   "background gradient differs across unmatched anchors");
        }

        // Darknet's (2 - truth.w*truth.h) box scale applies only to matched x/y/w/h loss, not
        // objectness/class. cx=cy=0 forces both offset targets to exactly 0 regardless of which
        // anchor/cell gets assigned, so the unscaled x/y gradient is exactly sigmoid(0)-0=0.5 and
        // any deviation from 0.5*box_scale isolates a missing or wrong scale factor.
        {
            auto scale_net = detector::Network::from_cfg((directory / "ignore-thresh.cfg").string());
            std::vector<std::vector<float>> output(scale_net.layers().size());
            std::vector<std::vector<float>> gradient(scale_net.layers().size());
            output[1].assign(12 * 16, 0.0f);
            gradient[1].assign(12 * 16, 0.0f);
            const float width = 0.5f, height = 0.4f;
            const float box_scale = 2.0f - width * height;
            std::vector<detector::Target> scale_targets{{0, 0.0f, 0.0f, width, height}};
            yolo_loss_and_gradient(scale_net, scale_targets, output, gradient);
            const std::size_t attrs = 6, area = 16;
            int match_anchor = -1;
            for (int a = 0; a < 2 && match_anchor < 0; ++a)
                if (std::abs(gradient[1][(static_cast<std::size_t>(a) * attrs + 4) * area] + 0.5f) < 1e-6f) match_anchor = a;
            expect(match_anchor >= 0, "no matched anchor found at cell 0 for box-scale test");
            float offset_x_grad = gradient[1][(static_cast<std::size_t>(match_anchor) * attrs + 0) * area];
            float offset_y_grad = gradient[1][(static_cast<std::size_t>(match_anchor) * attrs + 1) * area];
            expect(std::abs(offset_x_grad - 0.5f * box_scale) < 1e-5f, "x offset gradient missing Darknet's (2-area) box scale");
            expect(std::abs(offset_y_grad - 0.5f * box_scale) < 1e-5f, "y offset gradient missing Darknet's (2-area) box scale");
        }

        // SGD momentum: velocity starts at zero, so a trainer's first step() is identical to
        // plain SGD regardless of `momentum`, letting the first step recover g1 (the raw
        // gradient) from its own weight delta. The second step must then differ from a
        // momentum=0 trainer's by exactly lr*momentum*g1 (the carried-over velocity term).
        {
            auto momentum_cfg = (directory / "train-mini.cfg").string();
            std::vector<float> m_input{1.0f, 0.5f, 0.25f};
            std::vector<detector::Target> m_targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};
            const float lr = 0.01f;

            auto net0 = detector::Network::from_cfg(momentum_cfg);
            net0.initialize_random(3);
            detector::Trainer t0(net0, detector::LearningRateSchedule{lr}, 1, 0.0f, 0.0f);
            float w_initial = net0.layer(0).weights[0];
            t0.step({m_input}, {m_targets});
            float w_after1 = net0.layer(0).weights[0];
            t0.step({m_input}, {m_targets});
            float w0_after2 = net0.layer(0).weights[0];

            auto netm = detector::Network::from_cfg(momentum_cfg);
            netm.initialize_random(3);
            detector::Trainer tm(netm, detector::LearningRateSchedule{lr}, 1, 0.0f, 0.9f);
            tm.step({m_input}, {m_targets});
            float wm_after1 = netm.layer(0).weights[0];
            tm.step({m_input}, {m_targets});
            float wm_after2 = netm.layer(0).weights[0];

            expect(std::abs(w_after1 - wm_after1) < 1e-7f, "first momentum step must match plain SGD (zero initial velocity)");
            float g1 = (w_initial - w_after1) / lr;
            float expected_extra = lr * 0.9f * g1;
            float actual_extra = w0_after2 - wm_after2;
            expect(std::abs(actual_extra - expected_extra) < 1e-4f * std::max(1.0f, std::abs(expected_extra)),
                   "momentum did not carry velocity between steps as expected");
        }

        // Resume support (train_main.cpp's --resume-from): a Trainer seeded with a saved
        // `initial_iteration` must continue the LearningRateSchedule (burn-in then step-decay)
        // at exactly the iteration a prior, now-destroyed Trainer reached, not restart it at 0 -
        // the bug this feature fixes. Compares an uninterrupted 5-step run's LR-per-step sequence
        // and final weights against a simulated crash-and-resume: 2 steps, capture iteration()
        // (what the sidecar .step file would hold), then a *new* Trainer instance (as a relaunched
        // process creates) seeded with that iteration finishes the remaining 3 steps.
        {
            detector::LearningRateSchedule resume_schedule{0.1f, 2, 4.0f, {4}, {0.5f}};
            auto resume_cfg = (directory / "train-mini.cfg").string();
            std::vector<float> resume_input{1.0f, 0.5f, 0.25f};
            std::vector<detector::Target> resume_targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};

            auto net_cont = detector::Network::from_cfg(resume_cfg);
            net_cont.initialize_random(31);
            detector::Trainer cont(net_cont, resume_schedule, 1, 0.0f, 0.0f);
            std::vector<float> continuous_lrs;
            for (int i = 0; i < 5; ++i) {
                continuous_lrs.push_back(cont.current_learning_rate());
                cont.step({resume_input}, {resume_targets});
            }
            float continuous_weight = net_cont.layer(0).weights[0];

            auto net_resumed = detector::Network::from_cfg(resume_cfg);
            net_resumed.initialize_random(31);
            detector::Trainer first_half(net_resumed, resume_schedule, 1, 0.0f, 0.0f);
            for (int i = 0; i < 2; ++i) first_half.step({resume_input}, {resume_targets});
            int saved_iteration = first_half.iteration();
            expect(saved_iteration == 2, "iteration() did not count completed step() calls");

            detector::Trainer second_half(net_resumed, resume_schedule, 1, 0.0f, 0.0f, 0.99f, saved_iteration);
            std::vector<float> resumed_lrs(continuous_lrs.begin(), continuous_lrs.begin() + 2);
            for (int i = 2; i < 5; ++i) {
                resumed_lrs.push_back(second_half.current_learning_rate());
                second_half.step({resume_input}, {resume_targets});
            }
            for (int i = 0; i < 5; ++i)
                expect(std::abs(resumed_lrs[i] - continuous_lrs[i]) < 1e-9f,
                       "resumed LR sequence diverged from an uninterrupted run");
            expect(second_half.iteration() == cont.iteration(), "resumed iteration count did not reach the uninterrupted run's count");
            expect(std::abs(net_resumed.layer(0).weights[0] - continuous_weight) < 1e-5f,
                   "resumed training did not reproduce the uninterrupted run's final weights");
        }

        std::cout << "training smoke passed: first=" << first << " last=" << last << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training smoke failed: " << error.what() << '\n';
        return 1;
    }
}
