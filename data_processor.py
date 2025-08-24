# optimized_data_processor_v3.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class OptimizedBeerProcessAnomalyDetector:
    def __init__(self, beer_data_path='enhanced_beer_data.csv'):
        self.beer_data_path = beer_data_path
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
        self.quality_predictor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        self.anomaly_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1
        )

    def load_data(self):
        logger.info("Loading enhanced beer manufacturing data...")
        df = pd.read_csv(self.beer_data_path)
        logger.info(f"Dataset shape: {df.shape}")
        return df

    def prepare_features(self, df):
        logger.info("Preparing feature set for ML models...")
        process_features = [
            'malt_quantity_kg', 'water_volume_l', 'yeast_count_million',
            'mash_temperature_c', 'fermentation_temp_c', 'fermentation_ph',
            'pump_pressure_bar', 'cooling_efficiency_percent',
            'ambient_temperature_c', 'ambient_humidity_percent',
            'temp_deviation', 'ph_deviation', 'pressure_deviation'
        ]

        X_numeric = df[process_features].fillna(df[process_features].mean())

        top_styles = df['style'].value_counts().head(10).index
        df['style_simplified'] = df['style'].where(df['style'].isin(top_styles), 'Other')
        top_countries = df['country'].value_counts().head(8).index
        df['country_simplified'] = df['country'].where(df['country'].isin(top_countries), 'Other')
        X_categorical = pd.get_dummies(df[['style_simplified', 'country_simplified']], drop_first=True)

        X = pd.concat([X_numeric, X_categorical], axis=1)
        y_quality = df['abv']
        y_anomaly = df['process_anomaly']

        logger.info(f"Feature matrix shape: {X.shape}")
        return X, y_quality, y_anomaly, process_features

    def train_models(self, X, y_quality, y_anomaly):
        logger.info("Training machine learning models...")
        X_train, X_test, y_qual_train, y_qual_test, y_anom_train, y_anom_test = train_test_split(
            X, y_quality, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.quality_predictor.fit(X_train, y_qual_train)
        y_qual_pred = self.quality_predictor.predict(X_test)
        r2 = r2_score(y_qual_test, y_qual_pred)
        rmse = np.sqrt(mean_squared_error(y_qual_test, y_qual_pred))

        self.anomaly_classifier.fit(X_train_scaled, y_anom_train)
        y_anom_pred = self.anomaly_classifier.predict(X_test_scaled)
        acc = accuracy_score(y_anom_test, y_anom_pred)
        roc = roc_auc_score(y_anom_test, self.anomaly_classifier.predict_proba(X_test_scaled)[:, 1])
        class_report = classification_report(y_anom_test, y_anom_pred)

        self.anomaly_detector.fit(X_train_scaled[y_anom_train == 0])
        isolation_flags = self.anomaly_detector.predict(X_test_scaled)
        isolation_rate = (isolation_flags == -1).mean() * 100

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'accuracy': acc,
            'roc_auc': roc,
            'isolation_rate': isolation_rate,
            'classification_report': class_report
        }

        return X_train, X_test, y_qual_test, y_qual_pred, y_anom_test, y_anom_pred, metrics

    def create_visualizations(self, df, X_test, y_qual_test, y_qual_pred):
        logger.info("Generating visualizations...")
        plt.style.use('bmh')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        quality_counts = df['quality_grade'].value_counts()
        axes[0, 0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Quality Grade Distribution')

        top_styles = df['style'].value_counts().head(8).index
        style_anomaly = df[df['style'].isin(top_styles)].groupby('style')['process_anomaly'].mean()
        style_anomaly.plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Anomaly Rate by Beer Style')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[1, 0].scatter(df['fermentation_temp_c'], df['abv'],
                           c=df['process_anomaly'], cmap='RdYlGn_r', alpha=0.6)
        axes[1, 0].set_xlabel('Fermentation Temperature (°C)')
        axes[1, 0].set_ylabel('ABV (%)')
        axes[1, 0].set_title('Temperature vs ABV (Red = Anomaly)')

        axes[1, 1].scatter(y_qual_test, y_qual_pred, alpha=0.6)
        axes[1, 1].plot([y_qual_test.min(), y_qual_test.max()],
                        [y_qual_test.min(), y_qual_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual ABV')
        axes[1, 1].set_ylabel('Predicted ABV')
        axes[1, 1].set_title('ABV Prediction Performance')

        plt.tight_layout()
        plt.savefig('beer_manufacturing_analysis.png', dpi=150, bbox_inches='tight')  # Overwrite the same file
        plt.close(fig)
        logger.info("Visualization saved: beer_manufacturing_analysis.png")

    def save_models_and_data(self, df):
        logger.info("Saving models and processed data...")
        joblib.dump({
            'quality_predictor': self.quality_predictor,
            'anomaly_classifier': self.anomaly_classifier,
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler
        }, 'beer_manufacturing_models.pkl')
        df.to_csv('enhanced_beer_data.csv', index=False)  # overwrite the same dataset
        logger.info("Models and enhanced dataset saved.")

    def generate_summary_report(self, df, metrics):
        logger.info("Generating analysis summary report...")
        print("\n============================================================")
        print("BEER MANUFACTURING ANALYSIS SUMMARY")
        print("============================================================")
        print(f"Dataset: {len(df):,} batches analyzed")
        print(f"Unique Beer Styles: {df['style'].nunique()}")
        print(f"Countries Represented: {df['country'].nunique()}\n")

        print("Quality Distribution:")
        for grade in ['A', 'B', 'C']:
            count = (df['quality_grade'] == grade).sum()
            pct = count / len(df) * 100
            print(f"   Grade {grade}: {count:,} batches ({pct:.1f}%)")

        print(f"\nProcess Anomaly Rate: {df['process_anomaly'].mean():.1%}")
        print(f"Average ABV: {df['abv'].mean():.2f}%")
        print(f"Average Fermentation Temp: {df['fermentation_temp_c'].mean():.1f} °C")

        print("\nML Model Performance Metrics:")
        print(f" - ABV Prediction (R²) = {metrics['r2']:.4f}")
        print(f" - ABV Prediction (RMSE) = {metrics['rmse']:.4f}")
        print(f" - Anomaly Classification Accuracy = {metrics['accuracy']*100:.2f}%")
        print(f" - Anomaly Classification ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f" - Isolation Forest flagged = {metrics['isolation_rate']:.2f}% of batches")

        print("\nTop Styles with Highest Anomaly Rates:")
        style_anomaly = df.groupby('style')['process_anomaly'].agg(['mean', 'count'])
        style_anomaly = style_anomaly[style_anomaly['count'] >= 10]
        top_problem_styles = style_anomaly.nlargest(3, 'mean')
        for style, (rate, count) in top_problem_styles.iterrows():
            print(f"   {style[:30]} — {rate:.1%} anomaly rate ({count} batches)")
        print("============================================================")

    def run_analysis(self):
        df = self.load_data()
        X, y_quality, y_anomaly, _ = self.prepare_features(df)
        results = self.train_models(X, y_quality, y_anomaly)
        X_train, X_test, y_qual_test, y_qual_pred, y_anom_test, y_anom_pred, metrics = results
        self.create_visualizations(df, X_test, y_qual_test, y_qual_pred)
        self.save_models_and_data(df)
        self.generate_summary_report(df, metrics)
        return df, X, y_quality, y_anomaly, metrics


if __name__ == "__main__":
    detector = OptimizedBeerProcessAnomalyDetector('enhanced_beer_data.csv')
    df, X, y_quality, y_anomaly, metrics = detector.run_analysis()
