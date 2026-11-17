import {lazy, type ComponentType, type LazyExoticComponent} from "react";

export type CategoryId = "regression" | "optimization" | "classification" | "clustering" | "dimensionality" | "reinforcement";

export interface Category {
    id: CategoryId;
    title: string;
    blurb: string;
}

export interface PlaygroundMeta {
    slug: string;
    title: string;
    tagline: string;
    category: CategoryId;
    status: "live" | "soon";
}

export const CATEGORIES: Category[] = [
    {id: "regression", title: "Regression", blurb: "Predict a number"},
    {id: "optimization", title: "Optimization", blurb: "How models learn"},
    {id: "classification", title: "Classification", blurb: "Draw boundaries between classes"},
    {id: "clustering", title: "Clustering", blurb: "Find groups without labels"},
    {id: "dimensionality", title: "Dimensionality reduction", blurb: "Squeeze many dimensions into few"},
    {id: "reinforcement", title: "Reinforcement learning", blurb: "Learn by trial and error"},
];

export const PLAYGROUNDS: PlaygroundMeta[] = [
    {slug: "linear-regression", title: "Linear Regression", tagline: "Fit a line by shrinking a set of literal squares.", category: "regression", status: "live"},
    {slug: "gradient-descent", title: "Gradient Descent", tagline: "Race four optimizers across valleys, saddles and bumpy landscapes.", category: "optimization", status: "live"},
    {slug: "perceptron", title: "Perceptron", tagline: "A line that fixes itself, one mistake at a time.", category: "classification", status: "live"},
    {slug: "logistic-regression", title: "Logistic Regression", tagline: "Turn a hard boundary into calibrated probabilities.", category: "classification", status: "live"},
    {slug: "k-nearest-neighbors", title: "k-Nearest Neighbors", tagline: "Let the neighbors vote, and watch k trade noise for blur.", category: "classification", status: "live"},
    {slug: "neural-networks", title: "Neural Networks", tagline: "Train a network live and watch its layers bend space.", category: "classification", status: "live"},
    {slug: "k-means", title: "k-Means", tagline: "Assign, update, repeat: centroids hunting for clusters.", category: "clustering", status: "live"},
    {slug: "pca", title: "PCA", tagline: "Find the directions where the data actually varies.", category: "dimensionality", status: "live"},

    {slug: "svm", title: "Support Vector Machines", tagline: "The widest street between two classes, and kernels that bend it.", category: "classification", status: "live"},
    {slug: "decision-trees", title: "Decision Trees & Forests", tagline: "Carve the plane into boxes, one question at a time.", category: "classification", status: "live"},
    {slug: "naive-bayes", title: "Naive Bayes", tagline: "Classify with one Gaussian per class and Bayes' rule.", category: "classification", status: "soon"},
    {slug: "overfitting", title: "Overfitting & Regularization", tagline: "Crank up the polynomial degree, then rein it in.", category: "regression", status: "live"},
    {slug: "gaussian-processes", title: "Gaussian Processes", tagline: "Regression that knows where it is unsure.", category: "regression", status: "soon"},
    {slug: "gaussian-mixtures", title: "Gaussian Mixtures (EM)", tagline: "Soft clusters with ellipses that stretch to fit.", category: "clustering", status: "live"},
    {slug: "dbscan", title: "DBSCAN", tagline: "Clusters as dense regions, with noise left out.", category: "clustering", status: "soon"},
    {slug: "q-learning", title: "Q-Learning", tagline: "An agent learns a maze from rewards alone.", category: "reinforcement", status: "live"},
];

export const LIVE = PLAYGROUNDS.filter((p) => p.status === "live");

/** Live playgrounds in category order (used for previous/next navigation). */
export const ORDERED = CATEGORIES.flatMap((c) => LIVE.filter((p) => p.category === c.id));

export const PLAYGROUND_COMPONENTS: Record<string, LazyExoticComponent<ComponentType>> = {
    "linear-regression": lazy(() => import("./linear-regression/LinearRegression")),
    "gradient-descent": lazy(() => import("./gradient-descent/GradientDescent")),
    perceptron: lazy(() => import("./perceptron/Perceptron")),
    "logistic-regression": lazy(() => import("./logistic-regression/LogisticRegression")),
    "k-nearest-neighbors": lazy(() => import("./knn/Knn")),
    "neural-networks": lazy(() => import("./neural-networks/NeuralNetworks")),
    "k-means": lazy(() => import("./kmeans/KMeans")),
    pca: lazy(() => import("./pca/Pca")),
    svm: lazy(() => import("./svm/Svm")),
    "decision-trees": lazy(() => import("./trees/Trees")),
    "gaussian-mixtures": lazy(() => import("./gmm/Gmm")),
    "q-learning": lazy(() => import("./q-learning/QLearning")),
    overfitting: lazy(() => import("./overfitting/Overfitting")),
};

export function categoryOf(slug: string): Category | undefined {
    const p = PLAYGROUNDS.find((x) => x.slug === slug);
    return CATEGORIES.find((c) => c.id === p?.category);
}
