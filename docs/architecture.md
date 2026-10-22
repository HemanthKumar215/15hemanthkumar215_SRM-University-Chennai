# Architecture – Digital Twin for Type 2 Diabetes

## System Overview

```mermaid
graph LR
    subgraph Data_Sources["📊 Data Sources"]
        EHR["🏥 Synthetic EHR<br/>(Static Profile)"]
        CGM["📱 Synthetic CGM<br/>(Time-Series)"]
        WEAR["⌚ Wearable<br/>(HR, HRV, Steps, Sleep)"]
    end

    subgraph Ingestion["🔄 Ingestion & Storage"]
        GEN["Data Generator<br/>(500 patients × 30 days)"]
        STORE["Parquet / CSV<br/>Storage"]
    end

    subgraph Features["⚙️ Feature Engineering"]
        ROLL["Rolling Stats<br/>(Mean, Std, Slope)"]
        TIR["Time-in-Range<br/>(70-180 mg/dL)"]
        STATIC["Static Feature<br/>Encoding"]
    end

    subgraph Models["🧠 Modeling"]
        LGBM["LightGBM<br/>Baseline"]
        GRU["GRU Branch<br/>(Time-Series)"]
        MLP["MLP Branch<br/>(Static EHR)"]
        FUSION["Fusion Head<br/>(Concat → FC → σ)"]
    end

    subgraph Twin["🔮 Digital Twin"]
        STATE["Patient State<br/>(Sensor Buffer)"]
        PRED["Rolling<br/>Predictions"]
        WHATIF["What-If<br/>Simulator"]
    end

    subgraph Dashboard["📊 Dashboard"]
        DASH["Streamlit App<br/>(Doctor's View)"]
    end

    EHR --> GEN
    CGM --> GEN
    WEAR --> GEN
    GEN --> STORE
    STORE --> ROLL
    STORE --> TIR
    STORE --> STATIC
    ROLL --> LGBM
    TIR --> LGBM
    STATIC --> LGBM
    ROLL --> GRU
    STATIC --> MLP
    GRU --> FUSION
    MLP --> FUSION
    FUSION --> PRED
    STATE --> PRED
    PRED --> WHATIF
    PRED --> DASH
    WHATIF --> DASH

    style Data_Sources fill:#1a1a2e,stroke:#667eea,color:#fff
    style Ingestion fill:#16213e,stroke:#667eea,color:#fff
    style Features fill:#0f3460,stroke:#667eea,color:#fff
    style Models fill:#533483,stroke:#667eea,color:#fff
    style Twin fill:#e94560,stroke:#fff,color:#fff
    style Dashboard fill:#0f3460,stroke:#2ecc71,color:#fff
```

## Fusion Model Architecture

```mermaid
graph TB
    subgraph Input
        TS["Time-Series Input<br/>(288 × 5):<br/>Glucose, HR, HRV,<br/>Steps, Sleep"]
        ST["Static Input<br/>(26 features):<br/>Age, BMI, HbA1c,<br/>Meds, etc."]
    end

    subgraph GRU_Branch["GRU Branch"]
        G1["GRU Layer 1<br/>(hidden=64)"]
        G2["GRU Layer 2<br/>(hidden=64)"]
        LN["LayerNorm"]
    end

    subgraph MLP_Branch["MLP Branch"]
        M1["Linear(26 → 64) + ReLU"]
        BN1["BatchNorm1d + Dropout"]
        M2["Linear(64 → 32) + ReLU"]
        BN2["BatchNorm1d"]
    end

    CAT["Concatenate<br/>(64 + 32 = 96)"]

    subgraph Head["Classification Head"]
        H1["Linear(96 → 64) + ReLU"]
        DROP["Dropout(0.3)"]
        H2["Linear(64 → 1)"]
        SIG["Sigmoid → P(spike)"]
    end

    TS --> G1 --> G2 --> LN
    ST --> M1 --> BN1 --> M2 --> BN2
    LN --> CAT
    BN2 --> CAT
    CAT --> H1 --> DROP --> H2 --> SIG

    style GRU_Branch fill:#2c3e50,stroke:#3498db,color:#fff
    style MLP_Branch fill:#2c3e50,stroke:#9b59b6,color:#fff
    style Head fill:#2c3e50,stroke:#e74c3c,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant Gen as Data Generator
    participant Store as Data Store
    participant FE as Feature Engine
    participant Model as Fusion Model
    participant Twin as Patient Twin
    participant UI as Dashboard

    Gen->>Store: Generate EHR profiles (CSV)
    Gen->>Store: Generate CGM/wearable (Parquet)
    Store->>FE: Load patient data
    FE->>FE: Rolling stats, TIR, encoding
    FE->>Model: Feature matrix + labels
    Model->>Model: Train (patient-level split)
    Model->>Store: Save model checkpoint
    Store->>Twin: Load model + patient profile
    Twin->>Twin: Ingest sensor buffer
    Twin->>Twin: Run prediction
    Twin->>UI: Probability + risk level
    UI->>UI: Render charts & alerts
    Note over Twin,UI: What-if: modify buffer → re-predict → compare
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Patient-level split | Prevents data leakage (no patient in both train and test) |
| GRU over LSTM | Fewer parameters, comparable performance for this sequence length |
| 24-hour lookback | Captures circadian rhythm and dawn phenomenon |
| 2-hour prediction horizon | Actionable window for intervention |
| Weighted BCE loss | Handles class imbalance (spikes are minority class) |
| Forward-fill then zero-fill NaN | Preserves temporal continuity for GRU |
| Subsample negatives | Reduces training time while maintaining signal |
