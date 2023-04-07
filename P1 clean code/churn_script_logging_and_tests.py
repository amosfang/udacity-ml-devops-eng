import os
import logging
# import churn_library_solution as cls
import churn_library as cls
import joblib

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	logging.info('----- Testing import function -----')
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(import_data, perform_eda):
	'''
	test perform eda function
	'''
	logging.info('----- Testing eda function -----')
	# 1. Import data and perform eda
	try:
		df = import_data("./data/bank_data.csv")
		perform_eda(df)
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	# 2. Check for all eda image files
	file_pth = './images/eda/'

	# Can change the list of files as required
	file_lst = ['Churn_hist.png', 'Customer_Age_hist.png', 'Marital_Status_hist.png',
				'Total_Trans_Ct_histplot.png', 'eda_heatmap.png']

	found_lst = list()
	for file in file_lst:
		found_lst.append(os.path.isfile(file_pth + file))

	not_found_lst = [file for file, found in zip(file_lst, found_lst) if not found]

	try:
		assert not any(not_found_lst)
		logging.info("Testing eda_image files: SUCCESS")

	except AssertionError as err:
		for missing_file in not_found_lst:
			logging.error(f"Testing import_eda: {missing_file} wasn't found")
		raise err


def test_encoder_helper(import_data, encoder_helper, cat_columns):
	'''
	test encoder helper
	'''
	logging.info('----- Testing encoder helper function -----')

	cols_to_test = ['Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
					'Income_Category_Churn', 'Card_Category_Churn']
	try:
		# Import data
		df = import_data("./data/bank_data.csv")

		# Run encoder helper
		encoder_helper(df, cat_columns)

		# 1.Test - Check dataframe properties
		try:
			assert df.shape == (10127, 28)
			logging.info("Testing df shape: SUCCESS")
		except AssertionError as err:
			logging.error("Testing df : df should have 10127 rows and 28 columns."
						  f"There are {df.shape[0]} rows and {df.shape[1]} columns.")
			raise err
		# 2. Test - Check categorical columns
		try:
			assert sum(df.columns.isin(cols_to_test)) == len(cols_to_test)
			logging.info("Testing categorical columns: SUCCESS")
		except AssertionError as err:
			logging.error("Testing df : df should have 10127 rows and 28 columns."
						  f"There are {df.shape[0]} rows and {df.shape[1]} columns.")
			raise err
		# Test - Check if the Churn target column is present in the dataframe
		try:
			assert 'Churn' in df.columns
			logging.info("Test df : Churn column is present in the dataframe")
		except KeyError as err:
			logging.error("Testing import_data: The file doesn't not have the Churn target column")
			raise err
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

def test_perform_feature_engineering(import_data, perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	logging.info('----- Testing feature engineering function -----')
	keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
				 'Total_Relationship_Count', 'Months_Inactive_12_mon',
				 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
				 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
				 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
				 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
				 'Income_Category_Churn', 'Card_Category_Churn']
	try:
		# 1. Import data
		df = import_data("./data/bank_data.csv")

		# 2.Run perform feature engineering function
		X_train, X_test, y_train, y_test = perform_feature_engineering(df)

		try:
			assert all([col in X_train.columns for col in keep_cols])
			assert all([col in X_test.columns for col in keep_cols])
			logging.info("Testing of columns in X_train and X_test: SUCCESS")
		except AssertionError as err:
			logging.error("Testing of columns in X_train and X_test: Missing columns in X data")
			raise err

	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err


def test_train_models(import_data, perform_feature_engineering, train_models):
	'''
	test train_models
	'''
	logging.info('----- Testing train model function -----')
	# 1. Import data
	df = import_data("./data/bank_data.csv")
	# 2. Run perform feature engineering function
	X_train, X_test, y_train, y_test = perform_feature_engineering(df)
	# 3. Train models
	train_models(X_train, X_test, y_train, y_test)

	try:
		joblib.load("./models/rfc_model.pkl")
		joblib.load("./models/logistic_model.pkl")
		logging.info("Testing testing_models: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing train_models: Train model files were not found")
		raise err

if __name__ == "__main__":
	cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
				   'Income_Category', 'Card_Category']

	# 1. Running both logging and tests
	# Each of these tests can be run independently

	# test_import(cls.import_data)
	# test_eda(cls.import_data, cls.perform_eda)
	# test_encoder_helper(cls.import_data, cls.encoder_helper, cat_columns)
	# test_perform_feature_engineering(cls.import_data, cls.perform_feature_engineering)
	test_train_models(cls.import_data, cls.perform_feature_engineering, cls.train_models)

	# # 2. Running churn library functions without tests

	# df = cls.import_data('./data/bank_data.csv')
	# cls.perform_eda(df)
	# X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)
	# cls.train_models(X_train, X_test, y_train, y_test)





