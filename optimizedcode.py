import pandas as pd
import numpy as np
import keras
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

class TaxiFareModel:
    def __init__(self, df, features, label, learning_rate=0.001, epochs=20, batch_size=50):
        self.df = df
        self.features = features
        self.label = label
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.model_output = None
    
    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=1, input_shape=(len(self.features),)))
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                      loss="mean_squared_error",
                      metrics=[keras.metrics.RootMeanSquaredError()])
        self.model = model
    
    def train_model(self):
        features_data = self.df[self.features].values
        label_data = self.df[self.label].values
        history = self.model.fit(x=features_data, y=label_data, batch_size=self.batch_size, epochs=self.epochs)
        weights, bias = self.model.get_weights()
        epochs = history.epoch
        rmse = pd.DataFrame(history.history)["root_mean_squared_error"]
        self.model_output = (weights, bias, epochs, rmse)
    
    def plot_loss_curve(self, fig):
        _, _, epochs, rmse = self.model_output
        curve = px.line(x=epochs, y=rmse)
        curve.update_traces(line_color='#ff0000', line_width=3)
        fig.append_trace(curve.data[0], row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])
    
    def plot_model(self, fig):
        weights, bias, _, _ = self.model_output
        self.df['FARE_PREDICTED'] = bias[0]
        for index, feature in enumerate(self.features):
            self.df['FARE_PREDICTED'] += weights[index][0] * self.df[feature]
        
        if len(self.features) == 1:
            model_line = px.line(self.df, x=self.features[0], y='FARE_PREDICTED')
        else:
            z_name, y_name = "FARE_PREDICTED", self.features[1]
            z = [self.df[z_name].min(), (self.df[z_name].max() - self.df[z_name].min()) / 2, self.df[z_name].max()]
            y = [self.df[y_name].min(), (self.df[y_name].max() - self.df[y_name].min()) / 2, self.df[y_name].max()]
            x = [(z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0] for i in range(len(y))]
            plane = pd.DataFrame({'x': x, 'y': y, 'z': [z] * 3})
            model_line = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'], colorscale=[[0, '#89CFF0'], [1, '#FFDB58']]))
        
        fig.add_trace(model_line.data[0], row=1, col=2)
    
    def make_plots(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Curve", "Model Plot"), specs=[[{"type": "scatter"}, {"type": "surface" if len(self.features) > 1 else "scatter"}]])
        self.plot_loss_curve(fig)
        self.plot_model(fig)
        fig.show()
    
    def predict(self, batch_size=50):
        batch = self.df.sample(n=batch_size).copy()
        batch.reset_index(drop=True, inplace=True)
        predictions = self.model.predict_on_batch(batch[self.features].values)
        
        output = pd.DataFrame({
            "PREDICTED_FARE": predictions.flatten(),
            "OBSERVED_FARE": batch[self.label].values,
            "L1_LOSS": abs(predictions.flatten() - batch[self.label].values)
        })
        return output
    
    def run_experiment(self):
        self.build_model()
        self.train_model()
        self.make_plots()
        return self.model_output

# Usage:
# Load dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE']].copy()
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS'] / 60

# Define features and label
features = ['TRIP_MILES', 'TRIP_MINUTES']
label = 'FARE'

# Create model instance and run experiment
model = TaxiFareModel(training_df, features, label)
model.run_experiment()

# Make predictions
output = model.predict()
print(output.head())
