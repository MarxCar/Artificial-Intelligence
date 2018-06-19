import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import  LabelBinarizer, StandardScaler, CategoricalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#returns pandas array with the selected attributes of a dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values
#returns label binarizer back to pipeline friendly state
class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)





dataframe = pd.read_csv("datasets/housing/housing.csv")


housing_with_id = dataframe.reset_index() #adds index row
train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
#Makes different categories for income
dataframe["income_cat"] = np.ceil(dataframe["median_income"] / 1.5)
#merges income greater than 5 into category 5
dataframe["income_cat"].where(dataframe["income_cat"] < 5, 5.0, inplace=True)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #answer to life the universe and everything
for train_index, test_index in split.split(dataframe, dataframe["income_cat"]):
	
	strat_train_set = dataframe.loc[train_index]
	strat_test_set = dataframe.loc[test_index]
for set in (strat_train_set, strat_test_set):
	set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude",y="latitude", alpha=0.1, s=housing['population']/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


#plt.show()

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


#cant preprocess text
cat_attribs = ["ocean_proximity"]
housing_num = housing.drop(cat_attribs, axis=1)



num_attribs = list(housing_num)

num_pipeline = Pipeline([
	('selector', DataFrameSelector(num_attribs)),
	('imputer', SimpleImputer(strategy="median")),  #some rows of the toal_bedrooms attribute have no values
	#imputer calculates the median of each attribute and stores it in statistics_
	('std_scalar', StandardScaler()), #standardized the input by (x-mean)/range
])
cat_pipeline = Pipeline([
	('selector', DataFrameSelector(cat_attribs)),
	('label_binarizer', CategoricalEncoder()), #transform ocean proximity to a number using 1 hot encoding
])
full_pipeline = FeatureUnion(transformer_list=[
	("num_pipeline", num_pipeline),
	("cat_pipeline", cat_pipeline),
])


housing_prepared = full_pipeline.fit_transform(housing)

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_rmse = np.sqrt(mean_squared_error(housing_labels,forest_reg.predict(housing_prepared)))
print("Forest Error before Cross Reg: " + str(forest_rmse))

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10) #performs cross validation with k =10

#cross val takes a utility function and we use a cost function
h = np.sqrt(-scores)
print("Forest Mean and STD after Cross Reg:")
print(h.mean())
print(h.std())