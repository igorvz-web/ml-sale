/**
 * Линейная регрессия для предсказания следующей покупки
 * Реализация на чистом JavaScript без внешних зависимостей
 */

const LinearRegression = (function() {
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
     * Обучение модели линейной регрессии
     * @param {Array} data - массив объектов {features: [давность, частота, деньги, тренд], target: значение}
     * @param {Object} options - параметры обучения
     * @returns {Object} обученная модель
     */
    function trainLinearRegression(data, options = {}) {
        const learningRate = options.learningRate || 0.01;
        const iterations = options.iterations || 100;
        
        if (!data || data.length === 0) {
            return { weights: [], bias: 0, min: [], max: [], targetMin: 0, targetMax: 1, trained: false };
        }
        
        // Извлекаем признаки и целевые значения
        const X = data.map(d => d.features);
        const y = data.map(d => d.target);
        
        // Нормализуем признаки
        const { normalized: XNorm, min, max } = normalize(X);
        
        // Нормализуем целевые значения
        let targetMin = Math.min(...y);
        let targetMax = Math.max(...y);
        const yNorm = y.map(val => {
            const range = targetMax - targetMin;
            if (range === 0) return 0;
            return (val - targetMin) / range;
        });
        
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
                const pred = z;
                
                // Ошибка
                const error = pred - yNorm[i];
                
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
            targetMin,
            targetMax,
            trained: true
        };
    }

    /**
     * Предсказание следующего значения
     * @param {Array} features - массив признаков [давность, частота, деньги, тренд]
     * @param {Object} model - обученная модель
     * @returns {number} предсказанное значение
     */
    function predictNextValue(features, model) {
        if (!model.trained || !model.weights || model.weights.length === 0) {
            return 0; // Возвращаем 0 если модель не обучена
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
        
        // Вычисляем предсказание
        let z = model.bias;
        for (let i = 0; i < normalizedFeatures.length; i++) {
            z += model.weights[i] * normalizedFeatures[i];
        }
        
        // Денормализуем результат
        const range = model.targetMax - model.targetMin;
        return z * range + model.targetMin;
    }

    return {
        trainLinearRegression,
        predictNextValue,
        normalize
    };
})();

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LinearRegression;
}
