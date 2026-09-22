# 🚗 Insurance Claim Fraud Detection using Knowledge Distillation

A machine learning system for detecting fraudulent automobile insurance claims using **XGBoost**, **Knowledge Distillation**, and a lightweight **Neural Network**.

The project uses an XGBoost model as a teacher and transfers its learned behavior to a compact neural network student model. The resulting fraud probability is then used for automated claim routing into different processing categories.

---

## 📌 Project Overview

Insurance companies process a large number of claims, making manual investigation of every claim difficult and expensive.

This project proposes an end-to-end machine learning pipeline that:

1. Preprocesses insurance claim data.
2. Trains a **class-weighted XGBoost teacher model**.
3. Uses **Knowledge Distillation** to train a lightweight neural network.
4. Generates fraud probabilities.
5. Automatically routes claims based on fraud probability.

The project is specifically designed to study the effectiveness and limitations of knowledge distillation for highly imbalanced insurance fraud data.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │   Raw Insurance Claims  │
                 │       100,000 Records   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Data Preprocessing    │
                 │                         │
                 │ • Missing Value Handling│
                 │ • Duplicate Checking    │
                 │ • Scaling               │
                 │ • One-Hot Encoding      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   XGBoost Teacher Model │
                 │                         │
                 │ • Class Weighted       │
                 │ • Fraud Probability    │
                 └────────────┬────────────┘
                              │
                              │ Soft Labels
                              ▼
                 ┌─────────────────────────┐
                 │ Knowledge Distillation  │
                 │                         │
                 │ Hard Labels + Soft      │
                 │ Labels                  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Lightweight Neural      │
                 │ Network (Student)       │
                 │                         │
                 │ 128 → 64 → 32 → Output │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Fraud Probability     │
                 └────────────┬────────────┘
                              │
                              ▼
              ┌────────────────────────────────┐
              │      Automated Claim Routing   │
              ├────────────────────────────────┤
              │ ≤ 0.30  → Fast-Track Payout    │
              │ 0.30–0.70 → Standard Review    │
              │ > 0.70  → Fraud Investigation │
              └────────────────────────────────┘
