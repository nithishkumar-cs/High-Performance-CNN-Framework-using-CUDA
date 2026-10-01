#include "detector/trainer.hpp"
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
        float first = trainer.step(input, targets).loss;
        float last = first;
        for (int i = 0; i < 99; ++i) last = trainer.step(input, targets).loss;
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
        auto result = graph_trainer.step(std::vector<float>(graph.input_shape().elements(), 0.2f),
                                         {{0, 0.5f, 0.5f, 0.25f, 0.25f}});
        expect(std::isfinite(result.loss) && result.positives == 1 && result.updated,
               "branched graph (batch-normalize + route + upsample) training step");

        // Shortcut (residual) layer: forward+backward must run and reduce loss on a repeated target,
        // exercising fan-out gradient accumulation into both the shortcut's previous layer and branch.
        auto shortcut_net = detector::Network::from_cfg((directory / "shortcut-train.cfg").string());
        shortcut_net.initialize_random(11);
        detector::Trainer shortcut_trainer(shortcut_net, detector::LearningRateSchedule{0.05f}, 1, 0.0f);
        std::vector<float> shortcut_input{0.3f};
        std::vector<detector::Target> shortcut_targets{{0, 0.5f, 0.5f, 0.5f, 0.5f}};
        float shortcut_first = shortcut_trainer.step(shortcut_input, shortcut_targets).loss;
        float shortcut_last = shortcut_first;
        for (int i = 0; i < 49; ++i) shortcut_last = shortcut_trainer.step(shortcut_input, shortcut_targets).loss;
        expect(std::isfinite(shortcut_last) && shortcut_last < shortcut_first, "shortcut layer loss did not fall");

        // Minibatching: with batch_size 4, only every 4th step() call applies an SGD update, and
        // weights must stay frozen between those updates (pure gradient accumulation in between).
        auto batch_net = detector::Network::from_cfg((directory / "train-mini.cfg").string());
        batch_net.initialize_random(13);
        detector::Trainer batch_trainer(batch_net, detector::LearningRateSchedule{0.1f}, 4, 0.0f);
        auto weights_before = batch_net.layers()[0].weights;
        for (int i = 0; i < 3; ++i) {
            auto step_result = batch_trainer.step(input, targets);
            expect(!step_result.updated, "minibatch applied update before batch_size samples accumulated");
            expect(batch_net.layers()[0].weights == weights_before, "weights changed before minibatch boundary");
        }
        auto fourth = batch_trainer.step(input, targets);
        expect(fourth.updated, "minibatch did not apply update at batch_size boundary");
        expect(batch_net.layers()[0].weights != weights_before, "weights unchanged at minibatch boundary");

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
            return t.step(gc_input, gc_targets).loss;
        };
        auto analytic_grad = [&](int param) {
            auto net = detector::Network::from_cfg(bn_cfg);
            net.initialize_random(5);
            float before = param == 0 ? net.layer(0).scales[0] : param == 1 ? net.layer(0).biases[0] : net.layer(0).weights[0];
            float lr = 1e-4f;
            detector::Trainer t(net, detector::LearningRateSchedule{lr}, 1, 0.0f);
            t.step(gc_input, gc_targets);
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

        std::cout << "training smoke passed: first=" << first << " last=" << last << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training smoke failed: " << error.what() << '\n';
        return 1;
    }
}


