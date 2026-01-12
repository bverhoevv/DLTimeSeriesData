# Flight Delay Prediction with Deep Learning

This project applies deep learning models to time series flight delay data to predict expected delays for travelers departing from JFK Airport. The goal is to help passengers better understand delay patterns using real-world operational data.

## Dataset
We use a public Kaggle flight delays dataset containing airline, route, scheduling, aircraft, and delay information.
Since the data represents flights from a single day, the problem is framed as a short-horizon time series regression task rather than a seasonal forecasting problem.

To improve model performance, we evaluate:
-  1-hour delay averages
-  15-minute delay averages (higher resolution, reduced outliers)

## Models
The following architectures were implemented and compared:
1. Feedforward Neural Network
2. 1D CNN
3. RNN
4. LSTM
5. GRU
6. TCN
7. Transformer

All models were trained using the Adam optimizer and MSE loss, with architecture-specific hyperparameter tuning.

## Key Takeaways
- Finer 15-minute intervals significantly improved model performance
- Recurrent models benefited most from higher-resolution data
- TCN and Transformer models showed sensitivity to time aggregation
- Results reflect real-world variability in flight delays

## Tech Stack
Python, NumPy, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib
