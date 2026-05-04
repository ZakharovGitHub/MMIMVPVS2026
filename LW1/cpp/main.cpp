#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// блок 1: генерация данных

std::vector<double> generate_uniform(int n, double min, double max, std::mt19937& gen) {
    std::vector<double> result(n);
    std::uniform_real_distribution<double> dist(min, max);
    for (int i = 0; i < n; i++) {
        result[i] = dist(gen);
    }
    return result;
}

std::vector<double> generate_normal(int n, double mean, double stddev, std::mt19937& gen) {
    std::vector<double> result(n);
    std::normal_distribution<double> dist(mean, stddev);
    for (int i = 0; i < n; i++) {
        result[i] = dist(gen);
    }
    return result;
}

void generate_data(int n, std::vector<double>& X, std::vector<double>& Y,
    double x_min = -3.0, double x_max = 7.0,
    double noise_std = 0.3, unsigned int random_seed = 42) {
    std::mt19937 gen(random_seed);

    X = generate_uniform(n, x_min, x_max, gen);
    std::vector<double> noise = generate_normal(n, 0.0, noise_std, gen);

    Y.resize(n);
    for (int i = 0; i < n; i++) {
        Y[i] = 0.1 * (X[i] - 4) * std::cos(X[i]) + 0.5 * X[i] + noise[i];
    }
}

// блок 2: ядерная функция

double kernel_epanechnikov(double u) {
    if (std::abs(u) <= 1.0) {
        return 0.75 * (1.0 - u * u);
    }
    return 0.0;
}

double kernel_gaussian(double u) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * u * u);
}

double (*kernel)(double) = kernel_epanechnikov;

// блок 3: непараметрическая регрессия

double nadaraya_watson(double x_pred, const std::vector<double>& X_train,
    const std::vector<double>& Y_train, double c_param, int n_samples) {
    // параметр размытия по формуле (4.7.5)
    double h = c_param * std::pow(static_cast<double>(n_samples), -1.0 / 5.0);

    double numerator = 0.0;
    double denominator = 0.0;

    int train_size = static_cast<int>(X_train.size());
    for (int i = 0; i < train_size; i++) {
        double u = (x_pred - X_train[i]) / h;
        double k = kernel(u);
        numerator += k * Y_train[i];
        denominator += k;
    }

    if (denominator == 0.0) {
        double sum_y = 0.0;
        for (int i = 0; i < train_size; i++) {
            sum_y += Y_train[i];
        }
        return sum_y / static_cast<double>(train_size);
    }

    return numerator / denominator;
}

std::vector<double> predict_all(const std::vector<double>& X_pred,
    const std::vector<double>& X_train,
    const std::vector<double>& Y_train,
    double c_param) {
    int n_samples = static_cast<int>(X_train.size());
    int n_pred = static_cast<int>(X_pred.size());
    std::vector<double> Y_pred(n_pred);

    for (int i = 0; i < n_pred; i++) {
        Y_pred[i] = nadaraya_watson(X_pred[i], X_train, Y_train, c_param, n_samples);
    }

    return Y_pred;
}

// блок 4: скользящий экзамен для выбора параметра

double leave_one_out_cv(const std::vector<double>& X, const std::vector<double>& Y, double c_param) {
    int n = static_cast<int>(X.size());
    double sum_squared_errors = 0.0;

    for (int i = 0; i < n; i++) {
        std::vector<double> X_train, Y_train;
        X_train.reserve(n - 1);
        Y_train.reserve(n - 1);

        for (int j = 0; j < n; j++) {
            if (j != i) {
                X_train.push_back(X[j]);
                Y_train.push_back(Y[j]);
            }
        }

        double Y_pred = nadaraya_watson(X[i], X_train, Y_train, c_param, n - 1);
        double error = Y[i] - Y_pred;
        sum_squared_errors += error * error;
    }

    return sum_squared_errors / static_cast<double>(n);
}

void find_best_c(const std::vector<double>& X, const std::vector<double>& Y,
    double c_min, double c_max, double c_step,
    double& best_c, std::vector<double>& c_values, std::vector<double>& errors) {
    c_values.clear();
    errors.clear();

    int n_steps = static_cast<int>((c_max - c_min) / c_step) + 1;

    for (int i = 0; i < n_steps; i++) {
        double c = c_min + static_cast<double>(i) * c_step;
        double mse = leave_one_out_cv(X, Y, c);
        double h = c * std::pow(static_cast<double>(X.size()), -1.0 / 5.0);

        c_values.push_back(c);
        errors.push_back(mse);

        std::cout << "c = " << c << ", h = " << h << ", mse = " << mse << std::endl;
    }

    int best_idx = 0;
    int errors_size = static_cast<int>(errors.size());
    for (int i = 1; i < errors_size; i++) {
        if (errors[i] < errors[best_idx]) {
            best_idx = i;
        }
    }

    best_c = c_values[best_idx];
}

// блок 5: основная функция

int main() {
    std::setlocale(LC_ALL, "Russian");
    std::cout << "непараметрическая регрессия розенблатта-парзена" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << "\n[1] генерация данных..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> X, Y;
    generate_data(80, X, Y);

    auto generation_time = std::chrono::high_resolution_clock::now() - start_time;
    double generation_sec = std::chrono::duration<double>(generation_time).count();

    std::cout << "сгенерировано " << X.size() << " точек" << std::endl;
    std::cout << "время генерации: " << generation_sec << " сек" << std::endl;

    std::cout << "\n[2] поиск оптимального коэффициента c по формуле (4.7.5)..." << std::endl;
    std::cout << "перебор c от 0.1 до 2.0 с шагом 0.1" << std::endl;

    auto cv_start_time = std::chrono::high_resolution_clock::now();

    double best_c;
    std::vector<double> c_values, errors;
    find_best_c(X, Y, 0.1, 2.0, 0.1, best_c, c_values, errors);

    auto cv_time = std::chrono::high_resolution_clock::now() - cv_start_time;
    double cv_sec = std::chrono::duration<double>(cv_time).count();

    double min_error = errors[0];
    int errors_size = static_cast<int>(errors.size());
    for (int i = 1; i < errors_size; i++) {
        if (errors[i] < min_error) min_error = errors[i];
    }
    double best_h = best_c * std::pow(static_cast<double>(X.size()), -1.0 / 5.0);

    std::cout << "\nоптимальный коэффициент c: " << best_c << std::endl;
    std::cout << "соответствующий параметр размытия h: " << best_h << std::endl;
    std::cout << "минимальная mse: " << min_error << std::endl;
    std::cout << "время поиска: " << cv_sec << " сек" << std::endl;

    std::cout << "\n[3] построение модели регрессии..." << std::endl;

    auto pred_start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> X_smooth;
    for (double x = -3.0; x <= 7.0; x += 0.05) {
        X_smooth.push_back(x);
    }

    std::vector<double> Y_smooth = predict_all(X_smooth, X, Y, best_c);

    auto pred_time = std::chrono::high_resolution_clock::now() - pred_start_time;
    double pred_sec = std::chrono::duration<double>(pred_time).count();

    std::cout << "прогноз для " << X_smooth.size() << " точек" << std::endl;
    std::cout << "время прогнозирования: " << pred_sec << " сек" << std::endl;

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n" << std::endl;
    std::cout << "общее время выполнения: " << total_sec << " сек" << std::endl;

    std::cout << "\nинформация о модели:" << std::endl;
    std::cout << "  ядерная функция: епанечникова (формула 4.7.4)" << std::endl;
    std::cout << "  оптимальное c = " << best_c << ", h = " << best_h << std::endl;
    std::cout << "  размер выборки: " << X.size() << std::endl;

    return 0;
}
