from causal_learn.search.ConstraintBased.PC import pc
from causal_learn.utils.GraphUtil import GraphUtil
import networkx as nx
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from xgboost import XGBRegressor, XGBClassifier




def discover_causal_relationships(microbiome_data, clinical_variables):
    """
    Discover causal relationships between microbiome features and clinical outcomes
    """
    # 1. Create causal graph structure
    # Combine microbiome features and clinical variables
    data_matrix = np.column_stack([microbiome_data, clinical_variables])
    
    # 2. Apply PC algorithm with appropriate independence test
    # (Gaussian CI test for continuous data)
    cg = pc(data_matrix, 0.05, indep_test="gaussian")
    
    # 3. Extract and visualize the causal graph
    G = nx.DiGraph()
    edges = GraphUtil.graph2edges(cg.G)
    
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # 4. Identify key causal pathways to clinical outcomes
    # Find all paths from microbiome features to clinical outcomes
    paths_to_outcome = []
    for feature in microbiome_features:
        for outcome in clinical_outcomes:
            try:
                paths = nx.all_simple_paths(G, feature, outcome)
                paths_to_outcome.extend(paths)
            except nx.NetworkXNoPath:
                continue
    
    return G, paths_to_outcome

def estimate_causal_effects(data, treatment_taxa, outcome_variable):
    """
    Estimate causal effects of specific microbiome taxa on clinical outcomes
    using boosting-based causal inference methods
    """
    # 1. Implement Double Machine Learning with XGBoost
    # Separate features into treatment, outcome and confounders
    T = data[treatment_taxa]
    Y = data[outcome_variable]
    X = data.drop(columns=[treatment_taxa, outcome_variable])
    
    # 2. Create model with XGBoost as base learner
    model = CausalForestDML(
        model_t=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4),
        model_y=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4),
        n_estimators=1000,
        min_samples_leaf=5
    )
    
    # 3. Fit the model and estimate CATE (Conditional Average Treatment Effect)
    model.fit(Y, T, X=X)
    cate_estimates = model.effect(X)
    
    # 4. Calculate feature importance for heterogeneous treatment effects
    feature_importance = model.feature_importances_
    
    return cate_estimates, feature_importance