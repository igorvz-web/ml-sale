/**
 * Логистическая регрессия для предсказания вероятности покупки
 * Реализация на чистом JavaScript без внешних зависимостей
 */

const LogisticRegression = (function() {
    // Сигмоидальная функция
    function sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    // Нормализация данных (min-max scaling)
    function normalize(data) {
        if (!data || data.length === 0) return { normalized: [], min: [], max: [] };
        
        const numFeatures = data[0].length;
        const min = new Array(numFeatures).fill(Infinity);
        const max = new Array(numFeatures).fill(-Infinity);
        
        // Находим мин и макс для каждого признака
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < numFeatures; j++) {
                if (data[i][j] < min[j]) min[j] = data[i][j];
                if (data[i][j] > max[j]) max[j] = data[i][j];
            }
        }
        
        // Нормализуем
        const normalized = [];
        for (let i = 0; i < data.length; i++) {
            const row = [];
            for (let j = 0; j < numFeatures; j++) {
                const range = max[j] - min[j];
                if (range === 0) {
                    row.push(0);
                } else {
                    row.push((data[i][j] - min[j]) / range);
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
        
        if (!data || data.length === 0) {
            return { weights: [], bias: 0, min: [], max: [], trained: false };
        }
        
        // Извлекаем признаки и целевые значения
        const X = data.map(d => d.features);
        const y = data.map(d => d.target);
        
        // Нормализуем признаки
        const { normalized: XNorm, min, max } = normalize(X);
        
        const numFeatures = XNorm[0].length;
        const numSamples = XNorm.length;
        
        // Инициализируем веса и смещение
        let weights = new Array(numFeatures).fill(0);
        let bias = 0;
        
        // Градиентный спуск
        for (let iter = 0; iter < iterations; iter++) {
            let dw = new Array(numFeatures).fill(0);
            let db = 0;
            
            for (let i = 0; i < numSamples; i++) {
                // Предсказание
                let z = bias;
                for (let j = 0; j < numFeatures; j++) {
                    z += weights[j] * XNorm[i][j];
                }
                const pred = sigmoid(z);
                
                // Ошибка
                const error = pred - y[i];
                
                // Градиенты
                for (let j = 0; j < numFeatures; j++) {
                    dw[j] += error * XNorm[i][j];
                }
                db += error;
            }
            
            // Обновление весов
            for (let j = 0; j < numFeatures; j++) {
                weights[j] -= learningRate * (dw[j] / numSamples);
            }
            bias -= learningRate * (db / numSamples);
        }
        
        return {
            weights,
            bias,
            min,
            max,
            trained: true
        };
    }

    /**
     * Предсказание вероятности покупки
     * @param {Array} features - массив признаков [давность, частота, деньги, тренд]
     * @param {Object} model - обученная модель
     * @returns {number} вероятность от 0 до 1
     */
    function predictProbability(features, model) {
        if (!model.trained || !model.weights || model.weights.length === 0) {
            return 0.5; // Возвращаем нейтральную вероятность если модель не обучена
        }
        
        // Нормализуем входные данные
        const normalizedFeatures = [];
        for (let i = 0; i < features.length; i++) {
            const range = model.max[i] - model.min[i];
            if (range === 0) {
                normalizedFeatures.push(0);
            } else {
                normalizedFeatures.push((features[i] - model.min[i]) / range);
            }
        }
        
        // Вычисляем z
        let z = model.bias;
        for (let i = 0; i < normalizedFeatures.length; i++) {
            z += model.weights[i] * normalizedFeatures[i];
        }
        
        return sigmoid(z);
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
