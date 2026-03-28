# ML Module Integration Task

## Context

This project is a frontend BI dashboard (HTML + JS) that:

* loads Excel data
* processes clients
* calculates RFM, ABC, LTV
* has a heuristic probability score

We need to replace heuristic logic with real ML models (client-side only).

---

## GOAL

Add ML models:

1. Logistic Regression → purchase probability
2. Linear Regression → next purchase prediction

---

## REQUIREMENTS

### 1. Create new module

Create folder:

/ml/

Files:

* logistic.js
* regression.js
* mlService.js

---

### 2. Logistic Regression

Implement:

* sigmoid function
* gradient descent
* normalization

Functions:

trainLogisticRegression(data)
predictProbability(features, model)

Features:

* recency
* frequency
* monetary
* trend

Target:

* 0/1 (next purchase happened)

---

### 3. Linear Regression

Implement:

trainLinearRegression(data)
predictNextValue(input, model)

---

### 4. ML Service

Responsibilities:

* prepare dataset
* train models
* return predictions

Function:

runMLPipeline(clients)

---

### 5. Integration

After Excel load:

* call runMLPipeline()
* add fields:

client.mlProbability
client.mlPrediction

---

### 6. UI

Display:

* probability (%)
* predicted value

---

### 7. Performance

* fast training
* limit iterations
* normalize data

---

## IMPORTANT

* No backend
* Pure JS
* Do not break existing code
