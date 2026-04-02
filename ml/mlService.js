/**
 * ML Service - сервис для управления ML-моделями
 * Отвечает за подготовку данных, обучение моделей и возврат предсказаний
 */

const MLService = (function() {
    let logisticModel = null;
    let linearModel = null;

    /**
     * Подготовка обучающих данных из клиентов
     * @param {Array} clients - массив объектов клиентов
     * @returns {Object} подготовленные данные для обучения
     */
    function prepareTrainingData(clients) {
        const logisticData = [];
        const linearData = [];

        // Для каждого клиента создаем признаки и целевые значения
        for (let i = 0; i < clients.length; i++) {
            const client = clients[i];
            
            // Извлекаем признаки: давность, частота, деньги, тренд
            const recency = client.recency || 0;
            const frequency = client.frequency || 1;
            const monetary = client.monetary || 0;
            const trend = client.trend || 0;

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

            // Для линейной регрессии: target = следующая сумма покупки или дни до следующей покупки
            // Используем среднее значение как proxy
            const nextValue = (client.nextPurchaseValue !== undefined)
                ? client.nextPurchaseValue
                : (monetary * (hasNextPurchase ? 1.2 : 0.8));

            linearData.push({
                features: features,
                target: nextValue
            });
        }

        return { logisticData, linearData };
    }

    /**
     * Запуск полного ML пайплайна
     * @param {Array} clients - массив объектов клиентов
     * @returns {Object} результаты предсказаний для каждого клиента
     */
    function runMLPipeline(clients) {
        if (!clients || clients.length === 0) {
            return { predictions: [], modelsTrained: false };
        }

        console.log('[MLService] Запуск ML пайплайна для', clients.length, 'клиентов');

        // Подготавливаем данные
        const { logisticData, linearData } = prepareTrainingData(clients);

        // Обучаем логистическую регрессию
        logisticModel = LogisticRegression.trainLogisticRegression(logisticData, {
            learningRate: 0.1,
            iterations: 100
        });
        console.log('[MLService] Логистическая регрессия обучена', logisticModel.trained);

        // Обучаем линейную регрессию
        linearModel = LinearRegression.trainLinearRegression(linearData, {
            learningRate: 0.01,
            iterations: 100
        });
        console.log('[MLService] Линейная регрессия обучена', linearModel.trained);

        // Делаем предсказания для каждого клиента
        const predictions = clients.map(client => {
            const recency = client.recency || 0;
            const frequency = client.frequency || 1;
            const monetary = client.monetary || 0;
            const trend = client.trend || 0;

            const features = [recency, frequency, monetary, trend];

            // Предсказываем вероятность покупки
            const probability = LogisticRegression.predictProbability(features, logisticModel);

            // Предсказываем следующее значение
            const prediction = LinearRegression.predictNextValue(features, linearModel);

            return {
                clientId: client.id || client.client_id || null,
                mlProbability: probability,
                mlPrediction: prediction
            };
        });

        console.log('[MLService] Предсказания готовы для', predictions.length, 'клиентов');

        return {
            predictions,
            modelsTrained: logisticModel.trained && linearModel.trained
        };
    }

    /**
     * Получить предсказание для одного клиента
     * @param {Object} client - объект клиента
     * @returns {Object} предсказания
     */
    function predictForClient(client) {
        if (!logisticModel || !linearModel) {
            return {
                mlProbability: 0.5,
                mlPrediction: 0,
                modelsTrained: false
            };
        }

        const recency = client.recency || 0;
        const frequency = client.frequency || 1;
        const monetary = client.monetary || 0;
        const trend = client.trend || 0;

        const features = [recency, frequency, monetary, trend];

        const probability = LogisticRegression.predictProbability(features, logisticModel);
        const prediction = LinearRegression.predictNextValue(features, linearModel);

        return {
            mlProbability: probability,
            mlPrediction: prediction,
            modelsTrained: true
        };
    }

    /**
     * Проверка, обучены ли модели
     * @returns {boolean}
     */
    function isModelsTrained() {
        return (logisticModel && logisticModel.trained) && 
               (linearModel && linearModel.trained);
    }

    /**
     * Сброс моделей
     */
    function resetModels() {
        logisticModel = null;
        linearModel = null;
        console.log('[MLService] Модели сброшены');
    }

    return {
        runMLPipeline,
        predictForClient,
        isModelsTrained,
        resetModels,
        prepareTrainingData
    };
})();

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLService;
}
