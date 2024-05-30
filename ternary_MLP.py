# Multi-layer perceptron (MLP) machine learning model and ternary diagram plotter for cesium-lead-bromide (CsxPbyBrz) system.  
# This code maps the most likely phase from a Cs-Pb-Br reaction on the 16-ch reactor, based on its component mole ratio.
# Requirements:
#   1. An Excel sheet entitled "CsPbBr phase diagrams.xlsx" with sheets named "ratios vs phases", "true_X", and "true_Y"
#      placed in the same directory as this code.
#   2. The sheet "ratios vs phases" will give the mole ratios of Br, Cs, and Pb as columns, as well as list the Phase in a separate column (CsPbBr3, Cs4PbBr6, or coexistence of both species)
#   3. Additionally, Size and Symbol columns can be provided on this same sheet to aid in data visualization. Refer to the documentation for plotly.express.scatter_ternary.
#   4. The sheet "k_neighbors_X" will contain the observed Br-Cs-Pb mole ratios used for training the MLP. Copy-paste the observed and stoichiometric ratios from "ratios vs phases"
#   5. The sheet "k_neighbors_Y" will contain the observed Br-Cs-Pb phases used for training the MLP. Copy-paste the observed and stoichiometric phases from "ratios vs phases"
# This code's main outputs:
#   1. "ternary_grid.xlsx", an array representing the entire Cs-Pb-Br ternary mixing domain, and "support_vector_predictions.xlsx", the phase predictions based on the trained MLP. 
#      The fineness of the mesh can be easily adjusted.
#   2. A ternary plot containing observed, stoichiometric, and MLP-predicted phases represented as different colors. 
# author: R. Chairil, University of Southern California, Apr 2024
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import plotly.io as pio

# k-neighbors classifier and k-folds cross validation initialization
# note, data preprocessing/scaling is optional because the mole ratio data are all already normalized (i.e. sum to 1)
X = pd.read_excel("CsPbBr phase diagrams.xlsx", sheet_name='true_X')
y = pd.read_excel("CsPbBr phase diagrams.xlsx", sheet_name='true_Y')
rkf = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 42)
rkf.get_n_splits(X, y)

X_train = []
X_train = pd.DataFrame(X_train)
X_test = [] 
X_test = pd.DataFrame(X_test)
X_test.to_excel("X_test_mlp.xlsx") # initialize X_test and/or delete old contents
y_train = [] 
y_train = pd.DataFrame(y_train)
y_test = []
y_test = pd.DataFrame(y_test)
y_pred = []
y_pred = pd.DataFrame(y_pred)
y_pred.to_excel("train_result_mlp.xlsx") # initialize Y_pred and/or delete old contents

# call RepeatedKFold.split(X) to generate the indices used for the training and testing process
rkf_indices = pd.DataFrame(rkf.split(X))
rkf_indices.to_excel("rkf_indices_mlp.xlsx", index = True)

# grid search for best MLP parameters (no. of hidden layer neurons, activation fn, solver type, learning rate)
def MLP_GridSearch():
    best_paras = pd.Series()
    best_scores = pd.Series()
    params = [
    {
    'hidden_layer_sizes': [(50,),(60,),(75,),(100,),(200,),(300,)],
    'activation': ["logistic", "tanh", "relu"],
    'solver': ["lbfgs", "sgd", "adam"],
    'learning_rate': ['constant', 'adaptive']
    # 'max_iter': [(2000)] 
    }
    ]

    gridCV = GridSearchCV(estimator=MLPClassifier(), param_grid=params, scoring='accuracy', n_jobs=-1, cv=rkf)
    X_gs = X
    y_gs = y
    gridCV.fit(X_gs,np.array(y_gs).ravel())
    print(gridCV.best_params_)
    print('Optimal parameters:', gridCV.best_params_)
    print('Best Accuracy found:\n', gridCV.score(X_gs,y_gs))
    p_temp = pd.Series(gridCV.best_params_)
    best_paras = pd.concat([p_temp, best_paras], axis=1)
    s_temp = pd.Series(gridCV.score(X_gs,y_gs))
    best_scores = pd.concat([s_temp, best_scores], axis=1)

    best_paras.columns = np.arange(len(best_paras.columns))
    best_paras = best_paras.iloc[:,:-1]
    best_scores.columns = np.arange(len(best_scores.columns))
    best_scores = best_scores.iloc[:,:-1]
    
    return best_paras

# to optimize MLP hyperparameters - uncomment the two lines of code below; else, leave these commented out to reduce processing time
# param_filename = "mlp_GridSearch_results.xlsx"
# pd.DataFrame(MLP_GridSearch()).to_excel(param_filename, index = True)

# main MLP training module
# in a[x][y][z], [x] = 0 indicates training, [x] = 1 indicates testing
# [y] indicates the row number of a i.e. the fold iteration number. there will be a total of x = (n_splits)*(n_repeats) rows
# [z] indicates the respective test or train index. There will be a total of (total_n_of_data_pts)/(n_splits) testing z's and (total_n_of_data_pts)(1-1/(n_splits)) training z's (n_splits must be no less than 2)  
for n in range(rkf.get_n_splits(X, y)):
    for i in range(len(rkf_indices[0][n])):
        train_index = rkf_indices[0][n][i]
        if i == 1:
            X_train = X.iloc[[train_index]]
            y_train = y.iloc[[train_index]]
        else: 
            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train)
            X_train = pd.concat([X_train, X.iloc[[train_index]]])
            y_train = pd.concat([y_train, y.iloc[[train_index]]])
    print('X_train #', n+1, '=', X_train)
    print('y_train #', n+1, '=', y_train)

    for j in range(len(rkf_indices[1][n])):
        test_index = rkf_indices[1][n][j]
        if j == 1:
            X_test = X.iloc[[test_index]]
            y_test = y.iloc[[test_index]]
        else: 
            X_test = pd.DataFrame(X_test)
            y_test = pd.DataFrame(y_test)
            X_test = pd.concat([X_test, X.iloc[[test_index]]])
            y_test = pd.concat([y_test, y.iloc[[test_index]]])
    print('X_test #', n+1, '=', X_test)
    print('y_test #', n+1, '=', y_test)

    # scaling data is optional since all data have the same physical constraints
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train, y=y_train)
    # X_test = scaler.transform(X_test)

    models = []
    y_pred = []

    # train the MLP model per split
    MLP = MLPClassifier(hidden_layer_sizes=(75,), activation='relu', solver='lbfgs', learning_rate='constant', max_iter=2000, random_state=42)
    MLP.fit(X_train.values, np.array(y_train).ravel())
    models.append(MLP)
    y_pred.append(MLP.predict(X_test.values))

    # output the training and validation results
    fold_str = 'Fold no. ' + str(n+1)   
    y_pred = pd.DataFrame(y_pred).T
    with pd.ExcelWriter("train_result_mlp.xlsx", mode='a', if_sheet_exists = 'replace') as ywriter:
        y_pred.to_excel(ywriter, sheet_name = fold_str)

    X_test = pd.DataFrame(X_test)
    with pd.ExcelWriter("X_test_mlp.xlsx", mode='a', if_sheet_exists = 'replace') as xwriter: 
        X_test.to_excel(xwriter, sheet_name = fold_str)

    # estimate the overall model accuracy per fold and output to excel file 
    accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy for fold no.', n+1, '=', accuracy)
    accuracy = pd.DataFrame(accuracy, index =['Accuracy'], columns=[fold_str])

    if n == 0:
        accuracy_main = accuracy
    else:
        accuracy_main = pd.concat([accuracy_main, accuracy], axis = 1)
        accuracy_main.to_excel("accuracy_MLP.xlsx", sheet_name='accuracy')

    # convert back to array for plotting
    y_pred = np.array(y_pred)
    X_test = np.array(X_test)

# plot the learning curve
X_lc = X
y_lc = y
display = LearningCurveDisplay.from_estimator(MLP, X_lc.values, np.array(y_lc).ravel(), cv=25, shuffle=True, random_state=42, scoring='accuracy')

plt.title("Learning Curve")
plt.tight_layout()
plt.show()

# generate a mesh covering the entire Cs-Pb-Br space
a, b = np.mgrid[0:1:1000j, 0:1:1000j] # the higher the value in front of the 'j', the finer the mesh and the smoother the pseudocontours (but the longer the processing time)
mask = a + b <= 1
a = a[mask].ravel()
b = b[mask].ravel()
c = 1 - a - b

grid = (a,b,c)
grid = zip(*grid)
grid = pd.DataFrame(grid)
grid.columns = ['Br', 'Cs', 'Pb']
grid.to_excel("ternary_grid.xlsx", index=False)
X_u = pd.read_excel("ternary_grid.xlsx")

# apply the trained MLP model over the entire Cs-Pb-Br mixing space
y_u = MLP.predict(X_u.values)
y_u = pd.DataFrame(y_u)
y_u.to_excel("MLP_predictions.xlsx", index=False)

# main code for plotting ternary diagram
# to update, copy the values from the "ternary_grid.xlsx" output and place them in the Br-Cs-Pb columns of "CsPbBr phase diagrams.xlsx", sheet_name='ratios vs phases'
# then copy the values from the "support_vector_predictions.xlsx" output and place them in the phase column of "CsPbBr phase diagrams.xlsx", sheet_name='ratios vs phases'
# axes customization
def custAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': {'size': 30},
      'ticks': "outside",
      'tickangle': tickangle,
      'tickfont': {'size': 25},
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 7,
      'tickwidth': 7,
      'gridcolor': "#a9a9a9",
      'gridwidth': 1.5,
      'showline': True,
      'showgrid': True
    }

# adding the data to the plot
df = pd.read_excel("CsPbBr phase diagrams.xlsx", sheet_name='ratios vs phases')
fig = px.scatter_ternary(df, a="Br", b="Cs", c="Pb", hover_name="Phase",
    color="Plotted Phase", size='Size', symbol="Symbol", symbol_sequence=["star","circle","x","line-ew-open"],
    color_discrete_map = {"CsPbBr3": "red", "Cs4PbBr6": "blue", "CsPbBr3/Cs4PbBr6 coexistence":"violet", "neither phase/no reaction":"rgba(0,0,0,1)", 
    "CsPbBr3 ": "rgba(255, 0, 0, 0.01)", "Cs4PbBr6 ": "rgba(0, 0, 255, 0.01)", "CsPbBr3/Cs4PbBr6 coexistence ":"rgba(235, 165, 240, 0.01)", "neither phase/no reaction ":"rgba(211, 211, 211, 0.01)"}, opacity = 0.7) 
    # phase_ (with extra space) is for ML predicted zones; for ease of visualization, ML zones should be different contrasting colors and/or opacities
fig.update_layout(
    {'ternary': {
        'aaxis': custAxis('<b>''Br''</b>', 0),
        'baxis': custAxis('<b>''Cs''</b>', 55),
        'caxis': custAxis('<b>''Pb''</b>', -55)
    }},
    ternary_aaxis_dtick=0.1, ternary_baxis_dtick=0.1, ternary_caxis_dtick=0.1, 
    font_family="Helvetica",
    legend_title="",
    showlegend=False
    # legend=dict(
    #     yanchor="middle",
    #     y=0.95,
    #     xanchor="right",
    #     x=1.5,
    #     font=dict(family="Helvetica", size=55),
    #     itemsizing="constant",
    #     itemwidth=60
    #     )
    # )
    )
fig.show()

# write image to pdf - requires kaleido package (install using pip)
pio.write_image(fig, 'ternary_MLP.pdf',scale=1)