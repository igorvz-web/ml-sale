/**
 * Логистическая регрессия для предсказания вероятности покупки
 * Реализация на чистом JavaScript без внешних зависимостей
 */

const LogisticRegression = (function() {
    // Сигмоидальная функция с обработкой переполнений
    function sigmoid(z) {
        // Ограничиваем значение для избежания переполнения
        if (z > 100) return 1;
        if (z < -100) return 0;
        return 1 / (1 + Math.exp(-Math.min(z, 100)));
    }

    // Нормализация данных (min-max scaling) с улучшенной логикой
    function normalize(data) {
        if (!data || data.length === 0) return { normalized: [], min: [], max: [] };
        if (data.length < 2) return { normalized: data, min: [], max: [] };
        
        const numFeatures = data[0].length;
        const min = new Array(numFeatures).fill(Infinity);
        const max = new Array(numFeatures).fill(-Infinity);
        
        // Находим мин и макс для каждого признака
        for (let i = 0; i < data.length; i++) {
            if (!data[i] || data[i].length !== numFeatures) continue;
            for (let j = 0; j < numFeatures; j++) {
                const val = parseFloat(data[i][j]) || 0;
                if (val < min[j]) min[j] = val;
                if (val > max[j]) max[j] = val;
            }
        }
        
        // Нормализуем
        const normalized = [];
        for (let i = 0; i < data.length; i++) {
            if (!data[i]) continue;
            const row = [];
            for (let j = 0; j < numFeatures; j++) {
                const range = max[j] - min[j];
                const val = parseFloat(data[i][j]) || 0;
                if (range === 0) {
                    row.push(0.5);
                } else {
                    row.push((val - min[j]) / range);
                }
            }
            normalized.push(row);
        }
        
        return { normalized, min, max };
    }

    /**
     * Обучение модели логистической регрессии
     * @param {Array} data - массив объектов {features: [давность, частота, деньги, тренд], target: 0/1}
     * @param {Object} options - параметры обучения
     * @returns {Object} обученная модель
     */
    function trainLogisticRegression(data, options = {}) {
        const learningRate = options.learningRate || 0.1;
        const iterations = options.iterations || 100;
        
        // Валидация входных данных
        if (!data || data.length < 2) {
            console.warn('[LogisticRegression] Недостаточно данных для обучения (требуется >= 2)');
            return { weights: [], bias: 0, min: [], max: [], trained: false };
        }
        
        // Фильтруем некорректные данные
        const validData = data.filter(d => d && d.features && typeof d.target === 'number');
        if (validData.length < 2) {
            console.warn('[LogisticRegression] Недостаточно корректных данных');
            return { weights: [], bias: 0, min: [], max: [], trained: false };
        }
        
        // Извлекаем признаки и целевые значения
        const X = validData.map(d => d.features);
        const y = validData.map(d => (d.target > 0 ? 1 : 0));
        
        // Нормализуем признаки
        const { normalized: XNorm, min, max } = normalize(X);
        
        const numFeatures = XNorm[0].length;
        const numSamples = XNorm.length;
        
        // Инициализируем веса случайными значениями (поменьше)
        let weights = new Array(numFeatures).fill(0).map(() => Math.random() * 0.01);
        let bias = 0;
        
        let prevCost = Infinity;
        const tolerance = 1e-5;
        let converged = false;
        
        // Градиентный спуск с ранней остановкой
        for (let iter = 0; iter < iterations; iter++) {
            let dw = new Array(numFeatures).fill(0);
            let db = 0;
            let cost = 0;
            
            for (let i = 0; i < numSamples; i++) {
                // Предсказание
                let z = bias;
                for (let j = 0; j < numFeatures; j++) {
                    z += weights[j] * XNorm[i][j];
                }
                const pred = sigmoid(z);
                
                // Log-loss ошибка (для диагностики)
                const epsilon = 1e-7;
                cost += -(y[i] * Math.log(pred + epsilon) + (1 - y[i]) * Math.log(1 - pred + epsilon));
                
                // Ошибка
                const error = pred - y[i];
                
                // Градиенты
                for (let j = 0; j < numFeatures; j++) {
                    dw[j] += error * XNorm[i][j];
                }
                db += error;
            }
            
            cost /= numSamples;
            
            // Проверка сходимости
            if (Math.abs(prevCost - cost) < tolerance) {
                converged = true;
                console.log('[LogisticRegression] Сходимость достигнута на итерации', iter);
                break;
            }
            prevCost = cost;
            
            // Обновление весов с адаптивным learning rate
            const effectiveLR = learningRate / (1 + iter * 0.001); // Снижаем LR со временем
            for (let j = 0; j < numFeatures; j++) {
                weights[j] -= effectiveLR * (dw[j] / numSamples);
            }
            bias -= effectiveLR * (db / numSamples);
        }
        
        return {
            weights,
            bias,
            min,
            max,
            trained: true,
            converged
        };
    }

    /**
     * Предсказание вероятности покупки
     * @param {Array} features - массив признаков [давность, частота, деньги, тренд]
     * @param {Object} model - обученная модель
     * @returns {number} вероятность от 0 до 1
     */
    function predictProbability(features, model) {
        if (!model || !model.trained || !model.weights || model.weights.length === 0) {
            return 0.5; // Нейтральная вероятность если модель не обучена
        }
        
        // Валидация входных данных
        if (!features || !Array.isArray(features) || features.length !== model.weights.length) {
            console.warn('[LogisticRegression] Некорректные признаки для предсказания');
            return 0.5;
        }
        
        // Нормализуем входные данные
        const normalizedFeatures = [];
        for (let i = 0; i < features.length; i++) {
            const val = parseFloat(features[i]) || 0;
            const range = model.max[i] - model.min[i];
            if (range === 0) {
                normalizedFeatures.push(0.5);
            } else {
                normalizedFeatures.push((val - model.min[i]) / range);
            }
        }
        
        // Вычисляем z
        let z = model.bias || 0;
        for (let i = 0; i < normalizedFeatures.length; i++) {
            z += model.weights[i] * normalizedFeatures[i];
        }
        
        const probability = sigmoid(z);
        return Math.min(1, Math.max(0, probability)); // Ограничиваем [0, 1]
    }

    return {
        trainLogisticRegression,
        predictProbability,
        normalize,
        sigmoid
    };
})();

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LogisticRegression;
}
