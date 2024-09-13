import unittest
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class TestDiabetesModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup code executed once before all tests"""
        cls.seed = 42
        cls.dataset_path = "dataset/diabetes.csv"
        cls.metrics_path = "metrics.txt"
        cls.feature_importance_path = "feature_importance.png"
        cls.residuals_path = "residuals.png"
        # Load the dataset
        cls.df = pd.read_csv(cls.dataset_path)
        cls.y = cls.df.pop("Outcome")
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.df, cls.y, test_size=0.2,
                                                                            random_state=cls.seed)
        # Initialize model
        cls.model = RandomForestRegressor(max_depth=5, random_state=cls.seed)

    def test_data_loading(self):
        """Test if the data is loaded correctly"""
        self.assertFalse(self.df.empty, "Dataset is empty, failed to load properly.")
        self.assertIn('Outcome', self.y.name, "Target variable 'Outcome' is missing.")

    def test_model_training(self):
        """Test if the model is trained and scores are reasonable"""
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train) * 100
        test_score = self.model.score(self.X_test, self.y_test) * 100
        # Check if training and test scores are within a reasonable range
        self.assertGreater(train_score, 30, "Training score too low.")
        self.assertGreater(test_score, 30, "Test score too low.")

    def test_feature_importance(self):
        """Test if feature importance is calculated and file is created"""
        self.model.fit(self.X_train, self.y_train)
        importances = self.model.feature_importances_
        self.assertEqual(len(importances), self.X_train.shape[1], "Feature importances size mismatch.")

    def test_metrics_written(self):
        """Test if the metrics.txt file is written correctly"""
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train) * 100
        test_score = self.model.score(self.X_test, self.y_test) * 100
        with open(self.metrics_path, 'w') as f:
            f.write(f"Training variance explained: {train_score:.1f}%\n")
            f.write(f"Test variance explained: {test_score:.1f}%\n")
        # Check if the file exists
        self.assertTrue(os.path.exists(self.metrics_path), "metrics.txt file was not created.")
        # Read and validate the content
        with open(self.metrics_path, 'r') as f:
            lines = f.readlines()
            self.assertIn("Training variance explained", lines[0], "Training metrics not written correctly.")
            self.assertIn("Test variance explained", lines[1], "Test metrics not written correctly.")

    def test_plot_files_creation(self):
        """Test if the plots for feature importance and residuals are saved"""
        # Check for feature importance plot
        self.assertTrue(os.path.exists(self.feature_importance_path), "Feature importance plot was not saved.")
        # Check for residual plot
        self.assertTrue(os.path.exists(self.residuals_path), "Residuals plot was not saved.")

    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests"""
        # Remove created files
        if os.path.exists(cls.metrics_path):
            os.remove(cls.metrics_path)
        if os.path.exists(cls.feature_importance_path):
            os.remove(cls.feature_importance_path)
        if os.path.exists(cls.residuals_path):
            os.remove(cls.residuals_path)


if __name__ == '__main__':
    unittest.main()


