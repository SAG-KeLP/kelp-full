package it.uniroma2.sag.kelp.examples.demo.nystrom;

import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.dataset.selector.ExampleSelector;
import it.uniroma2.sag.kelp.data.dataset.selector.RandomExampleSelector;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.examples.demo.qc.QuestionClassification;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.MultiEpochLearning;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.dcd.DCDLoss;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.learningalgorithm.classification.scw.SCWType;
import it.uniroma2.sag.kelp.learningalgorithm.classification.scw.SoftConfidenceWeightedClassification;
import it.uniroma2.sag.kelp.linearization.nystrom.NystromMethod;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

public class NystromExampleMain {

	public static final String LINEAR_REP_NAME = "lin";

	public static void main(String[] args) throws Exception {

		String trainFilePath = "src/main/resources/qc/train_5500.coarse.klp.gz";
		String testFilePath = "src/main/resources/qc/TREC_10.coarse.klp.gz";

		/*
		 * Select the number of landmarks
		 */
		int landmarkSize = 500;
		int randomSeed = 0;

		/*
		 * Select a specific kernel from the existing ones:
		 * - bow: a Linear Kernel applied to a boolean Bag-of-Word vector
		 * - stk: a Subset Tree Kernel 
		 * - ptk: a Partial Tree Kernel
		 * - sptk: a Smoothed Partial Tree Kernel
		 * - csptk: Compositionally Smoothed Partial Tree Kernel
		 */
		String kernelId = "csptk";

		SimpleDataset trainDataset = new SimpleDataset();
		trainDataset.populate(trainFilePath);
		SimpleDataset testDataset = new SimpleDataset();
		testDataset.populate(testFilePath);

		Kernel kernel = QuestionClassification.getQCKernelFunction(trainDataset, testDataset, kernelId);
		/**
		 * Select Landmark
		 */
		ExampleSelector landmarkSelector = new RandomExampleSelector(landmarkSize, randomSeed);
		List<Example> landmarks = landmarkSelector.select(trainDataset);

		/**
		 * Apply the Nystrom Method
		 */
		NystromMethod nystromMethod = new NystromMethod(landmarks, kernel);

		SimpleDataset linearizedTrainDataset = nystromMethod.getLinearizedDataset(trainDataset, LINEAR_REP_NAME);
		SimpleDataset linearizedTestDataset = nystromMethod.getLinearizedDataset(testDataset, LINEAR_REP_NAME);

		/**
		 * Evaluating the batch and online classifier over the linearized
		 * vectors
		 */
		float accuracy;
		LearningAlgorithm learningAlgorithm;

		linearizedTrainDataset.setSeed(randomSeed);

		learningAlgorithm = new DCDLearningAlgorithm(5, 5, DCDLoss.L2, false, 30, LINEAR_REP_NAME);
		accuracy = evaluateClassifier(linearizedTrainDataset, linearizedTestDataset, learningAlgorithm);
		System.out.println("Batch Learning Accuracy:\t" + accuracy);

		learningAlgorithm = new SoftConfidenceWeightedClassification(null, SCWType.SCW_II, 0.90f, 2, 2, false,
				LINEAR_REP_NAME);
		learningAlgorithm = new MultiEpochLearning(2, learningAlgorithm);
		accuracy = evaluateClassifier(linearizedTrainDataset, linearizedTestDataset, learningAlgorithm);
		System.out.println("Online Learning Accuracy:\t" + accuracy);
	}

	private static float evaluateClassifier(SimpleDataset linearizedTrainDataset, SimpleDataset linearizedTestDataset,
			LearningAlgorithm learningAlgorithm) {
		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(learningAlgorithm);
		ovaLearner.setLabels(linearizedTrainDataset.getClassificationLabels());

		ovaLearner.learn(linearizedTrainDataset);

		Classifier f = ovaLearner.getPredictionFunction();

		MulticlassClassificationEvaluator ev = new MulticlassClassificationEvaluator(
				linearizedTrainDataset.getClassificationLabels());

		for (Example e : linearizedTestDataset.getExamples()) {
			ClassificationOutput p = f.predict(e);
			ev.addCount(e, p);
		}
		float accuracy = ev.getAccuracy();
		return accuracy;
	}

}
