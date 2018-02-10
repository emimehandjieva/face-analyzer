import numpy as np

class FuzzyDecisionTree:
    MEAN_THRESHOLD = 0.3

    def __init__(self,dataset):
        #self.dataset = dataset
        self.dataset = self.transform_dataset(dataset)
        self.tree = self.build_tree(100,1)

    def transform_dataset(selfself,dataset):
        transformed_dataset = []
        for item in dataset:
            transformed_dataset.append([item,1])
        return transformed_dataset

    def build_tree(self, max_depth, min_size):
        root = self.get_split(self.dataset)
        self.split(root, max_depth, min_size, 1)
        return root

    def gini_index(self,groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Split a dataset based on an attribute and an attribute value
    def test_split(self,index, value, dataset):
        left, right = list(), list()
        #calculate beta
        minValue = min(row[0][index] for row in dataset)
        maxValue = max(row[0][index] for row in dataset)
        beta = maxValue-minValue/3
        for row in dataset:
            if row[0][index] < value:
                membership_left = self.get_membership_function_left(row[0][index],beta,value)
                # calculate membership of element to current node
                row[1] = min(row[1],membership_left)
                left.append(row)
            else:
                membership_right = self.get_membership_function_right(row[0][index], beta, value)
                # calculate membership of element to current node
                row[1] = min(row[1], membership_right)
                right.append(row)
        return left, right

     # Calculate membership function of each element in set to current splitting
    def get_membership_function_left(self, value, beta, splitter):
        if splitter - beta >= value:
            return 1
        else:
            return (0.5 + (splitter - value) / beta)

    def get_membership_function_right(self, value, beta, splitter):
        if splitter - beta <= value:
            return 1
        else:
            return (0.5 + (value - splitter) / beta)

    # Select the best split point for a dataset
    def get_split(self,dataset):
        class_values = list(set(row[0][-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[0][index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a terminal node value
    def to_terminal(self,group):
        # TODO: return not a maximum , avg at least, but most likely fuzzy addition
        valueSum = sum([row[0][-1]*row[1] for row in group])
        membershipSum = sum([row[1] for row in group])
        return valueSum/membershipSum

    # Create child splits for a node or make terminal
    def split(self,node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def classify(self, row):
        return self.predict(self.tree, row)

    def predict(self, node, row):
        # TODO : include fuzzyness
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
