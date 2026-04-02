/**
 * ML Service - сервис для управления ML-моделями
 * Отвечает за подготовку данных, обучение моделей и возврат предсказаний
 */

const MLService = (function() {
    let logisticModel = null;
    let linearModel = null;
    let lastPredictions = {};

    /**
     * Подготовка обучающих данных из клиентов
     * @param {Array} clients - массив объектов клиентов
     * @returns {Object} подготовленные данные для обучения
     */
    function prepareTrainingData(clients) {
        const logisticData = [];
        const linearData = [];

        if (!Array.isArray(clients)) return { logisticData, linearData };

        // Для каждого клиента создаем признаки и целевые значения
        for (let i = 0; i < clients.length; i++) {
            const client = clients[i];
            if (!client) continue;
            
            // Извлекаем признаки: давность, частота, деньги, тренд
            const recency = parseFloat(client.recency) || 0;
            const frequency = parseFloat(client.frequency) || 1;
            const monetary = parseFloat(client.monetary) || 0;
            const trend = parseFloat(client.trend) || 0;

            const features = [recency, frequency, monetary, trend];

            // Для логистической регрессии: target = 1 если клиент активен (частота > 1), иначе 0
            // В реальных данных это должно быть основано на фактической следующей покупке
            const hasNextPurchase = (client.hasNextPurchase !== undefined) 
                ? (client.hasNextPurchase ? 1 : 0)
                : (frequency > 1 ? 1 : 0);

            logisticData.push({
                features: features,
                target: hasNextPurchase
            });

            // Для линейной регрессии: target = следующая сумма покупки
            // Используем среднее значение как proxy или реальное значение если есть
            const nextValue = parseFloat(client.nextPurchaseValue) !== undefined
                ? parseFloat(client.nextPurchaseValue)
                : (monetary > 0 ? monetary * (hasNextPurchase ? 1.2 : 0.8) : 100);

            linearData.push({
                features: features,
                target: Math.max(0, nextValue) // Убеждаемся, что значение >= 0
            });
        }

        console.log('[MLService] Данные подготовлены:', logisticData.length, 'образцов');
        return { logisticData, linearData };
    }

    /**
     * Запуск полного ML пайплайна
     * @param {Array} clients - массив объектов клиентов
     * @returns {Object} результаты предсказаний для каждого клиента
     */
    function runMLPipeline(clients) {
        if (!Array.isArray(clients) || clients.length === 0) {
            console.warn('[MLService] Пустой массив клиентов');
            return { predictions: [], modelsTrained: false };
        }

        console.log('[MLService] Запуск ML пайплайна для', clients.length, 'клиентов');

        try {
            // Подготавливаем данные
            const { logisticData, linearData } = prepareTrainingData(clients);

            if (logisticData.length < 2 || linearData.length < 2) {
                console.warn('[MLService] Недостаточно данных для обучения');
                return { predictions: [], modelsTrained: false };
            }

            // Обучаем логистическую регрессию
            logisticModel = LogisticRegression.trainLogisticRegression(logisticData, {
                learningRate: 0.1,
                iterations: 150
            });
            console.log('[MLService] ✓ Логистическая регрессия обучена, сходимость:', logisticModel.converged);

            // Обучаем линейную регрессию
            linearModel = LinearRegression.trainLinearRegression(linearData, {
                learningRate: 0.01,
                iterations: 150
            });
            console.log('[MLService] ✓ Линейная регрессия обучена');

            // Делаем предсказания для каждого клиента
            const predictions = clients.map((client, index) => {
                if (!client) return null;
                
                const recency = parseFloat(client.recency) || 0;
                const frequency = parseFloat(client.frequency) || 1;
                const monetary = parseFloat(client.monetary) || 0;
                const trend = parseFloat(client.trend) || 0;

                const features = [recency, frequency, monetary, trend];

                try {
                    // Предсказываем вероятность покупки
                    const probability = LogisticRegression.predictProbability(features, logisticModel);

                    // Предсказываем следующее значение
                    const prediction = LinearRegression.predictNextValue(features, linearModel);

                    return {
                        index: index,  // Индекс клиента в массиве
                        clientName: client.name || client.client_name || `Клиент #${index}`,
                        mlProbability: Math.round(probability * 10000) / 10000, // Округляем до 4 знаков
                        mlPrediction: Math.round(prediction * 100) / 100,       // Округляем до 2 знаков
                        features: features
                    };
                } catch (err) {
                    console.error('[MLService] Ошибка при предсказании для клиента', index, err);
                    return null;
                }
            }).filter(p => p !== null);

            // Сохраняем последние предсказания
            lastPredictions = {};
            predictions.forEach(pred => {
                lastPredictions[pred.index] = pred;
            });

            console.log('[MLService] ✓ Предсказания готовы для', predictions.length, 'клиентов');

            return {
                predictions,
                modelsTrained: logisticModel && logisticModel.trained && linearModel && linearModel.trained,
                logisticAccuracy: logisticModel.trained ? 'обучена' : 'не обучена',
                linearAccuracy: linearModel.trained ? 'обучена' : 'не обучена'
            };
        } catch (err) {
            console.error('[MLService] Критическая ошибка в ML пайплайне:', err);
            return { predictions: [], modelsTrained: false };
        }
    }

    /**
     * Получить предсказание для одного клиента
     * @param {Object} client - объект клиента
     * @returns {Object} предсказания
     */
    function predictForClient(client) {
        if (!logisticModel || !logisticModel.trained || !linearModel || !linearModel.trained) {
            return {
                mlProbability: 0.5,
                mlPrediction: 0,
                modelsTrained: false
            };
        }

        if (!client) {
            return {
                mlProbability: 0.5,
                mlPrediction: 0,
                modelsTrained: false
            };
        }

        try {
            const recency = parseFloat(client.recency) || 0;
            const frequency = parseFloat(client.frequency) || 1;
            const monetary = parseFloat(client.monetary) || 0;
            const trend = parseFloat(client.trend) || 0;

            const features = [recency, frequency, monetary, trend];

            const probability = LogisticRegression.predictProbability(features, logisticModel);
            const prediction = LinearRegression.predictNextValue(features, linearModel);

            return {
                mlProbability: Math.round(probability * 10000) / 10000,
                mlPrediction: Math.round(prediction * 100) / 100,
                modelsTrained: true
            };
        } catch (err) {
            console.error('[MLService] Ошибка при предсказании для клиента:', err);
            return {
                mlProbability: 0.5,
                mlPrediction: 0,
                modelsTrained: false
            };
        }
    }

    /**
     * Получить последние предсказания
     * @returns {Object} сохраненные предсказания по индексам
     */
    function getLastPredictions() {
        return lastPredictions;
    }

    /**
     * Проверка, обучены ли модели
     * @returns {boolean}
     */
    function areModelsTrained() {
        return (logisticModel && logisticModel.trained) && 
               (linearModel && linearModel.trained);
    }

    /**
     * Сброс моделей
     */
    function resetModels() {
        logisticModel = null;
        linearModel = null;
        lastPredictions = {};
        console.log('[MLService] Модели сброшены');
    }

    /**
     * Получить статус моделей
     * @returns {Object} информация о моделях
     */
    function getStatus() {
        return {
            logisticTrained: logisticModel && logisticModel.trained,
            linearTrained: linearModel && linearModel.trained,
            predictionsCount: Object.keys(lastPredictions).length,
            logisticWeights: logisticModel ? logisticModel.weights.length : 0,
            linearWeights: linearModel ? linearModel.weights.length : 0
        };
    }

    return {
        runMLPipeline,
        predictForClient,
        areModelsTrained,
        resetModels,
        prepareTrainingData,
        getLastPredictions,
        getStatus
    };
})();

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLService;
}
