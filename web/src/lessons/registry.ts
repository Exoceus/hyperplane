import {lazy, type ComponentType, type LazyExoticComponent} from "react";

export type TrackId = "foundations" | "classification" | "unsupervised";

export interface Track {
    id: TrackId;
    title: string;
    blurb: string;
}

export interface LessonMeta {
    slug: string;
    title: string;
    tagline: string;
    track: TrackId;
    minutes?: number;
    status: "live" | "soon";
}

export const TRACKS: Track[] = [
    {id: "foundations", title: "Foundations", blurb: "Fitting a line, and the idea of walking downhill that trains every model after it."},
    {id: "classification", title: "Classification", blurb: "Drawing boundaries between classes, from a single line to networks that bend space."},
    {id: "unsupervised", title: "Unsupervised learning", blurb: "Finding structure in data that comes without any labels."},
];

export const LESSONS: LessonMeta[] = [
    {
        slug: "linear-regression",
        title: "Linear Regression",
        tagline: "Why “least squares” means literally shrinking a set of squares.",
        track: "foundations",
        minutes: 10,
        status: "live",
    },
    {
        slug: "gradient-descent",
        title: "Gradient Descent",
        tagline: "Roll downhill on a loss surface, and see why the step size decides everything.",
        track: "foundations",
        minutes: 12,
        status: "live",
    },
    {
        slug: "perceptron",
        title: "The Perceptron",
        tagline: "The first learning machine: a line that fixes itself, one mistake at a time.",
        track: "classification",
        minutes: 12,
        status: "live",
    },
    {
        slug: "logistic-regression",
        title: "Logistic Regression",
        tagline: "Turn a hard yes/no boundary into calibrated probabilities.",
        track: "classification",
        minutes: 10,
        status: "live",
    },
    {
        slug: "k-nearest-neighbors",
        title: "k-Nearest Neighbors",
        tagline: "Classify by polling the neighbors, and watch k trade noise for blur.",
        track: "classification",
        minutes: 8,
        status: "live",
    },
    {
        slug: "neural-networks",
        title: "Neural Networks",
        tagline: "Stack simple neurons and watch them bend space until a line is enough.",
        track: "classification",
        minutes: 15,
        status: "live",
    },
    {
        slug: "k-means",
        title: "k-Means Clustering",
        tagline: "Two alternating steps that find clusters without being told what to look for.",
        track: "unsupervised",
        minutes: 8,
        status: "live",
    },
    {
        slug: "pca",
        title: "Principal Component Analysis",
        tagline: "Find the directions in which your data actually varies.",
        track: "unsupervised",
        minutes: 10,
        status: "live",
    },
    {slug: "svm", title: "Support Vector Machines", tagline: "Out of all separating lines, pick the one with the widest street.", track: "classification", status: "soon"},
    {slug: "kernel-trick", title: "The Kernel Trick", tagline: "Lift data into higher dimensions where a plane can split it.", track: "classification", status: "soon"},
    {slug: "decision-trees", title: "Decision Trees", tagline: "Carve the plane into boxes, one question at a time.", track: "classification", status: "soon"},
    {slug: "backpropagation", title: "Backpropagation", tagline: "The chain rule, flowing backwards through a network.", track: "foundations", status: "soon"},
    {slug: "regularization", title: "Overfitting & Regularization", tagline: "Why a model that memorizes generalizes badly, and how to stop it.", track: "foundations", status: "soon"},
    {slug: "gaussian-mixtures", title: "Gaussian Mixtures & EM", tagline: "Soft clusters, fit by alternately guessing and refining.", track: "unsupervised", status: "soon"},
];

export const LIVE_LESSONS = LESSONS.filter((l) => l.status === "live");

export const LESSON_COMPONENTS: Record<string, LazyExoticComponent<ComponentType>> = {
    "linear-regression": lazy(() => import("./linear-regression/LinearRegression")),
    "gradient-descent": lazy(() => import("./gradient-descent/GradientDescent")),
    perceptron: lazy(() => import("./perceptron/Perceptron")),
    "logistic-regression": lazy(() => import("./logistic-regression/LogisticRegression")),
    "k-nearest-neighbors": lazy(() => import("./knn/Knn")),
    "neural-networks": lazy(() => import("./neural-networks/NeuralNetworks")),
    "k-means": lazy(() => import("./kmeans/KMeans")),
    pca: lazy(() => import("./pca/Pca")),
};

export function lessonNumber(slug: string): number {
    return LIVE_LESSONS.findIndex((l) => l.slug === slug) + 1;
}
