import unittest
import sys
import os
from sklearn.preprocessing import LabelEncoder

from src.ML_FLOW import KaggleMLFlow

class TestKaggleMLFlow(unittest.TestCase):
    
    def setUp(self):
        self.flow = KaggleMLFlow()
        self.path = "C:/Users/jesco/OneDrive - Universidad Santo Tomás/Documentos/Python/T3"
        self.train_file = os.path.join(self.path, "train.csv")
        self.test_file = os.path.join(self.path, "test.csv")
        self.apply_feature_engineering = True
        self.model_rank = 4
        
    def test_data_loading(self):
        df, prueba, df_id, prueba_id = self.flow.load_data(download_path=self.path)
        self.assertIsNotNone(df, "La carga del conjunto de entrenamiento falló.")
        self.assertIsNotNone(prueba, "La carga del conjunto de prueba falló.")
        self.assertGreater(len(df), 0, "El conjunto de entrenamiento está vacío.")
        self.assertGreater(len(prueba), 0, "El conjunto de prueba está vacío.")
        print("Prueba de carga de datos exitosa.")
        
    def test_column_extraction(self):
        df, prueba, df_id, prueba_id = self.flow.load_data(download_path=self.path)
        ct = ['Gender', 'Displaced', 'Educational special needs', 'Debtor', 'Scholarship holder', 
              'International', 'Marital status', 'Nacionality', "Mother's occupation", "Father's occupation"]
        base_modelo, cuantitativas_bm, categoricas_bm, le = self.flow.get_columns(df, prueba, ct)
        
        self.assertIsNotNone(base_modelo, "La extracción de columnas falló.")
        self.assertGreater(len(cuantitativas_bm), 0, "No se identificaron columnas cuantitativas.")
        self.assertGreater(len(categoricas_bm), 0, "No se identificaron columnas categóricas.")
        self.assertIsInstance(le, LabelEncoder, "El objeto LabelEncoder no fue creado correctamente.")
        print("Prueba de extracción de columnas exitosa.")
    
    def test_complete_workflow_execution(self):
        result = self.flow.run_entire_work_flow(self.path, self.apply_feature_engineering, self.model_rank)
        self.assertTrue(result['success'], "El pipeline no se ejecutó satisfactoriamente.")
        print("Ejecución completa del pipeline exitosa.")
        
        train_accuracy = result['accuracy_train']
        test_accuracy = result['accuracy_test']

        self.assertGreater(train_accuracy, 0.6, "Posible underfitting: baja precisión en entrenamiento.")
        self.assertGreater(test_accuracy, 0.3, "Posible underfitting: baja precisión en prueba.")

        self.assertLess(abs(train_accuracy - test_accuracy), 0.6, "Posible overfitting: gran diferencia entre entrenamiento y prueba.")
        print("Prueba de underfitting/overfitting exitosa.")
        
    def test_pipeline_without_feature_engineering(self):
        self.apply_feature_engineering = False
        result = self.flow.run_entire_work_flow(self.path, self.apply_feature_engineering, self.model_rank)
        self.assertTrue(result['success'], "El pipeline sin ingeniería de características no se ejecutó satisfactoriamente.")
        print("Prueba del pipeline sin ingeniería de características exitosa.")
    
    def tearDown(self):
        # Eliminar archivos descargados después de la prueba
        if os.path.exists(self.train_file):
            os.remove(self.train_file)
            print(f"Archivo {self.train_file} eliminado.")
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            print(f"Archivo {self.test_file} eliminado.")

if __name__ == '__main__':
    unittest.main()
