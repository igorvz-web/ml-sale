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
        
        // Валидация входных данных
        if (!data || data.length < 2) {
            console.warn('[LinearRegression] Недостаточно данных для обучения (требуется >= 2)');
            return { weights: [], bias: 0, min: [], max: [], targetMin: 0, targetMax: 1, trained: false };
        }
        
        // Фильтруем некорректные данные
        const validData = data.filter(d => 
            d && d.features && typeof d.target === 'number' && isFinite(d.target)
        );
        if (validData.length < 2) {
            console.warn('[LinearRegression] Недостаточно корректных данных');
            return { weights: [], bias: 0, min: [], max: [], targetMin: 0, targetMax: 1, trained: false };
        }
        
        // Извлекаем признаки и целевые значения
        const X = validData.map(d => d.features);
        const y = validData.map(d => d.target);
        
        // Нормализуем признаки
        const { normalized: XNorm, min, max } = normalize(X);
        
        // Нормализуем целевые значения
        let targetMin = Math.min(...y);
        let targetMax = Math.max(...y);
        
        // Избегаем деления на ноль
        if (targetMin === targetMax) {
            targetMax = targetMin + 1;
        }
        
        const yNorm = y.map(val => {
            const range = targetMax - targetMin;
            return (val - targetMin) / range;
        });
        
        const numFeatures = XNorm[0].length;
        const numSamples = XNorm.length;
        
        // Инициализируем веса случайными значениями
        let weights = new Array(numFeatures).fill(0).map(() => Math.random() * 0.01);
        let bias = 0;
        
        let prevMSE = Infinity;
        const tolerance = 1e-6;
        
        // Градиентный спуск
        for (let iter = 0; iter < iterations; iter++) {
            let dw = new Array(numFeatures).fill(0);
            let db = 0;
            let mse = 0;
            
            for (let i = 0; i < numSamples; i++) {
                // Предсказание
                let z = bias;
                for (let j = 0; j < numFeatures; j++) {
                    z += weights[j] * XNorm[i][j];
                }
                const pred = z;
                
                // MSE для мониторинга
                const error = pred - yNorm[i];
                mse += error * error;
                
                // Градиенты
                for (let j = 0; j < numFeatures; j++) {
                    dw[j] += error * XNorm[i][j];
                }
                db += error;
            }
            
            mse /= numSamples;
            
            // Проверка сходимости
            if (Math.abs(prevMSE - mse) < tolerance) {
                console.log('[LinearRegression] Сходимость достигнута на итерации', iter, 'MSE:', mse.toFixed(6));
                break;
            }
            prevMSE = mse;
            
            // Обновление весов с адаптивным learning rate
            const effectiveLR = learningRate / (1 + iter * 0.0005);
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
        if (!model || !model.trained || !model.weights || model.weights.length === 0) {
            return 0; // Возвращаем 0 если модель не обучена
        }
        
        // Валидация входных данных
        if (!features || !Array.isArray(features) || features.length !== model.weights.length) {
            console.warn('[LinearRegression] Некорректные признаки для предсказания');
            return 0;
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
        
        // Вычисляем предсказание (нормализованное)
        let z = model.bias || 0;
        for (let i = 0; i < normalizedFeatures.length; i++) {
            z += model.weights[i] * normalizedFeatures[i];
        }
        
        // Ограничиваем [0, 1]
        z = Math.min(1, Math.max(0, z));
        
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
