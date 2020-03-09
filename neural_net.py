import pandas as pd
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

breast_cancer = load_breast_cancer()
df_features = preprocessing.scale(breast_cancer.data)  # with so many features, values get high.  Scale for activation function
df_features = pd.DataFrame(data=df_features, columns=breast_cancer.feature_names)

df_target = pd.DataFrame(data=breast_cancer.target, columns=['target'])

df = pd.merge(df_features, df_target, left_index=True, right_index=True, how='left')


# remove outliers with z score
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# find correlated variables
corr = df.corr()

ax = sns.heatmap(corr,
                 vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20, 220, n=200),
                 square=True
                 )

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

corr = corr.unstack().sort_values(ascending=False).drop_duplicates()

# Create correlation matrix
corr_df = corr.to_frame()

# Select upper triangle of correlation matrix
upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop features showing dependency
df.drop(to_drop, axis=1, inplace=True)

# split into test/train (or don't be lazy and import train_test_split)
train = round(len(df.index) * 0.8)
test = round(len(df.index) * 0.2)

df_train = df.head(train)
df_test = df.tail(test)

# create functions to automated neural network process
# this is all to just easily optimize your # of layers and nodes between each layer with grid_search
def tanh(x):  # activation function
    return np.tanh(x)


def tanh_p(x):  # optimize yer activation function
    return 1.0 - np.tanh(x)**2


def neural_network(layered_nodes):
    # how many layers do you have?
    max_layer = []
    for ln in layered_nodes:
        max_layer.append(ln[0])

    max_layer = max(max_layer)  

    num_weights = []  # how many weights are needed to get to each layer 
    prev = 1
    for ln in layered_nodes:
        if ln[0] > 1:
            num_weights.append(
                ln[1] * prev)  # current node inputs * node outputs 
        if ln[0] == max_layer:
            num_weights.append(ln[1])  # if it's the last layer, just need a weight for each node going to output
        prev = ln[1]

    # now assign your weights the amount of weights you need.
    # layer one weights look like w1_1, w1_2, etc..
    # layer three would look like w3_1, w3_2
    weights = {}
    biases = {}
    layer = 1
    for n in num_weights:
        for j in range(1, n + 1):
            weights['w{0}'.format(str(layer) + str(j))] = np.random.random()
        layer += 1

    # get a bias for each layer
    # should look like b1, b2, b3
    for b in range(1, max_layer + 1):
        biases['b{0}'.format(b)] = np.random.random()
    return weights, biases, num_weights


# now that you have your weights and 'structure' for your neural network, feed through nodes
def feedforward(weights, biases, num_weights, inputs):
    # starting at the beginning of the dictionary weights -
    # split the dictionary into input weights 
    # store the remaining dictionary in hidden_weights to separate out later
    pos = 0
    input_weights = []
    input_bias = list(biases.values())
    hidden_weights = []
    all_nodes = []
    for n in num_weights:  # split weights into layer groups
        temp = dict(list(weights.items())[pos: (pos + n)])
        for k, v in temp.items():
            if pos == 0:
                input_weights.append(v)
            elif pos > 0:
                hidden_weights.append(v)
        pos += n

    # feedforward just inputs here
    start = 0
    end = len(inputs)
    activated_nodes = []
    h1_nodes = int((len(input_weights)) / len(inputs))  # total layer 1 weights / # input nodes = # output nodes 
    for n in range(1, h1_nodes + 1):  # for each output node, take dot product of all inputs + bias
        in_weights = input_weights[
                     start: start + end]  # gives the first 3 weights for node 1, then next 3 weights for node 2, etc..
        products = []
        node = 0
        for num1, num2 in zip(inputs, in_weights):
            products.append(num1 * num2)

        for j in products:
            node += j

        node = node + input_bias[0]
        node = tanh(node)
        activated_nodes.append(node)  # nodes are activated and ready to use in next feedforward step
        all_nodes.append(node)  # keeping track of node outputs for back prop step later on

        start += end
    del input_bias[0]  # delete bias after using it

    # feedforward hidden layers now from node results
    for n in num_weights[1:]:  # first val always goes to input layer, take values starting at second indices
        start = 0
        end = len(activated_nodes)
        output_nodes = int(n / len(activated_nodes))  # num_weights / num_inputs  (from prev nodes)
        h_activated = []  # temp location for recursion

        for j in range(1, output_nodes + 1):  # for each node, take dot product of all previous nodes + bias
            in_weights = hidden_weights[start: start + end]  # split weights for this layer
            hidden_products = []
            h_node = 0
            for num1, num2 in zip(activated_nodes, in_weights):  # run and hide cause you just used 2 nested loops 
                hidden_products.append(num1 * num2)

            for k in hidden_products:
                h_node += k

            h_node += input_bias[0]
            h_node = tanh(h_node)
            h_activated.append(h_node)
            all_nodes.append(h_node)  # for keeping track for back propagation step

            del hidden_weights[start: start + end]  # delete weights after using them
        del input_bias[0]  # then delete bias after using it
        activated_nodes = h_activated
    return activated_nodes[0], all_nodes  


# now that you've fed through, optimize your weights with respect to your cost function using back propagation
def back_propagation(inputs, prediction, target, weights, num_weights, biases, learning_rate, layered_nodes, all_nodes):
    cost = (prediction - target) ** 2  # just here to plot costs over iters, should see a decrease
    # optimize parameters by deriving with respect to cost (flex power rule here - lol, so lame)
    dcost_dpred = 2 * (prediction - target)
    dpred_dz = tanh_p(prediction)
    dcost_dz = dcost_dpred * dpred_dz

    # derive biases 
    for key, value in biases.items():
        biases[key] = value - (learning_rate * dcost_dz * 2)

    # for the inputs
    in_vals = []
    n = int(num_weights[0] / len(inputs))  # 12 weights / 3 inputs = 4 nodes (4 outputs)
    for r in range(n):  # for all outputs, append to in_vals
        for x in inputs:
            in_vals.append(x)

    # for everything past the first input layer
    # basically transforming node_outputs to grow vertically for deriving
    # example: for the first layer node1 = 1, node2 = 2, node3 = 3, node 4 = 4.
    # instead of having a list of [1, 2, 3, 4], you need to multiply occurence of each number by number of prev nodes
    # so should transform to [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] since it came from 3 inputs
    how_many_nodes = []
    for node in layered_nodes[1:]:
        how_many_nodes.append(node[1])

    how_many_replications = []
    for num1, num2 in zip(num_weights[1:], how_many_nodes):
        how_many_replications.append(num1 / num2)

    node_transformation = [list(a) for a in zip(how_many_nodes, how_many_replications)]
    transformed = []

    for val in node_transformation:
        end = val[0]
        for j in range(0, int(val[1])):
            transformed.append(all_nodes[0: end])

    transformed = [item for sublist in transformed for item in
                   sublist]  # OKAY. Now were really ready to back propagate

    start_dict = 0
    start_list = 0
    for n in num_weights:
        temp = dict(list(weights.items())[start_dict:start_dict + n])
        for key, val in temp.items():
            if start_dict == 0:  # the first layer here
                weights[key] = val - (learning_rate * dcost_dz * in_vals[start_list])
            if start_dict > 1:  # anything beyond the first layer
                weights[key] = val - (learning_rate * dcost_dz * transformed[start_list])
        start_dict += n
        start_list += 1

    # now take care of optimizing the last layers that lead to the output
    last_layer = num_weights[-1] * - 1
    pos = last_layer
    for key, val in dict(list(weights.items())[last_layer:]).items():
        if pos < 0:
            weights[key] = val - (learning_rate * dcost_dz * all_nodes[pos])
            pos += 1

    return cost, biases, weights


# create neural network 'back bone'
features = [column for column in df.columns if column != 'target']
nn_layers = [[1, len(features)], [2, 15], [3, 7], [4, 3]]
nn_weights, nn_biases, nn_num_weights = neural_network(nn_layers)

# feedforward & back propagate
predicted_outcomes = []
output_nodes_outcomes = []
costs = []
for index, row in df_train.iterrows():
    input_list = row.tolist()
    pred_train, output_nodes = feedforward(nn_weights, nn_biases, nn_num_weights, input_list[:-1])
    nn_cost, nn_biases, nn_weights = back_propagation(input_list[:-1], pred_train, input_list[-1], nn_weights, nn_num_weights, nn_biases, 0.01, nn_layers, output_nodes)
    predicted_outcomes.append(pred_train)
    output_nodes_outcomes.append(output_nodes)
    costs.append(nn_cost)


# test model for accuracy
test_predictions = []
for index, row in df_test.iterrows():
    input_list = row.tolist()
    pred_test, output_nodes = feedforward(nn_weights, nn_biases, nn_num_weights, input_list[:-1])
    test_predictions.append(round(pred_test))

model_score = accuracy_score(df_test['target'].values.tolist(), test_predictions) # yay 92%! 

