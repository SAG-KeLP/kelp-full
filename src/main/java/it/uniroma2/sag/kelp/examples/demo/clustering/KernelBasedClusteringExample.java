/*
 * Copyright 2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.examples.demo.clustering;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.clustering.ClusterList;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansEngine;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansExample;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;

/**
 * This class contains an example of the usage of the Kernel-based clustering.
 * The clustering engine implements the Kernel Based K-means described in:
 * 
 * Brian Kulis, Sugato Basu, Inderjit Dhillon, and Raymond Mooney.
 * Semi-supervised graph clustering: a kernel approach. Machine Learning,
 * 74(1):1-22, January 2009.
 * 
 * The source code are provided by Kelp in the kernel-clustering project.
 * 
 * The dataset used in this example is the IRIS dataset. It can be downloaded
 * from: https://archive.ics.uci.edu/ml/datasets/Iris
 * 
 * @author Danilo Croce
 * 
 */
public class KernelBasedClusteringExample {

	public static void main(String[] args) throws Exception {
		// Number of clusters computed by the Kernel-based K-means algorithm
		int K = 6;
		// Number of iteration of the Kernel-based K-means algorithm
		int tMax = 10;
		// Load the dataset
		SimpleDataset dataset = new SimpleDataset();
		dataset.populate("src/main/resources/iris_dataset/iris_dataset.klp");

		// The representation considered from the algorithm
		String representationName = "0";

		// Initialize the kernel function
		Kernel kernel = new LinearKernel(representationName);
		// Initialize the cache
		FixIndexKernelCache kernelCache = new FixIndexKernelCache(
				dataset.getNumberOfExamples());
		kernel.setKernelCache(kernelCache);

		// Initializing the clustering engine
		KernelBasedKMeansEngine clusteringEngine = new KernelBasedKMeansEngine(
				kernel, K, tMax);

		// Example of serialization of the engine via JSON
		ObjectSerializer serializer = new JacksonSerializerWrapper();
		System.out.println(serializer.writeValueAsString(clusteringEngine));

		// Run the clustering
		ClusterList clusterList = clusteringEngine.cluster(dataset);

		System.out.println("\n==================");
		System.out.println("Resulting clusters");
		System.out.println("==================\n");
		// Writing the resulting clusters and cluster members
		for (Cluster cluster : clusterList) {
			for (ClusterExample clusterMember : cluster.getExamples()) {
				float dist = ((KernelBasedKMeansExample) clusterMember)
						.getDist();
				System.out.println(dist + "\t" + cluster.getLabel() + "\t"
						+ clusterMember.getExample());
			}
			System.out.println();
		}
	}
}
