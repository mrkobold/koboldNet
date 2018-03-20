package koboldNet;

import javafx.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * KoboldNet
 */
@SuppressWarnings("DanglingJavadoc")
public class KoboldNet {

	private List<Double[][]> weights = new ArrayList<>();
	private List<Double[]> activations = new ArrayList<>();

	// 1 input, 1 output, n-2 hidden layers
	private int numLayers = 4;
	private int numNeuronsOnLayers[] = new int[]{784, 16, 16, 10};

	// learning rate alfa
	private static double α = 0.5;

	/**
	 * Initializes this neu-net's weights
	 */
	@SuppressWarnings("unused")
	public void initialize() {
		for (int i = 0; i < numLayers - 1; i++) {
			Double[][] weightsI = initializeWeights(i);
			weights.add(i, weightsI);
		}
	}

	public void trainNetwork(List<Double[]> inputs, List<Double[]> labels) {
		Double[] features = inputs.get(0);
		forwardPass(features);
		Pair<List<Double[][]>, Double> bpResult = backwardsPropagation(labels.get(0));
		updateWeights(bpResult.getKey());
		Double cost = bpResult.getValue();

		for (int i = 1; i < inputs.size(); i++) {
			features = inputs.get(i);
			forwardPass(features);
			bpResult = backwardsPropagation(labels.get(i));
			updateWeights(bpResult.getKey());
			double deltaCost = cost - bpResult.getValue();
			/** maybe this learn rate update rule has to be updated */
			α -= deltaCost > 0 ? deltaCost * deltaCost * 2 : 0;
			α = Math.max(α, 0.00001);
			if (i % 50 == 0) {
				System.out.println("α = " + α + " cost = " + cost);
			}
			cost = bpResult.getValue();
		}
	}

	private void updateWeights(List<Double[][]> weightDerivatives) {
		for (int i = 0; i < numLayers - 1; i++) {
			Double[][] weightDerivativeMatrix = weightDerivatives.get(i);
			for (int j = 0; j < weightDerivativeMatrix.length; j++) {
				for (int k = 0; k < weightDerivativeMatrix[j].length; k++) {
					weights.get(i)[j][k] -= weightDerivativeMatrix[j][k] * α;
				}
			}
		}
	}

	/**
	 * Updates the weights of the neu-net using backpropagation
	 *
	 * @param expectedOutputs the outputs that are expected
	 */
	public Pair<List<Double[][]>, Double> backwardsPropagation(Double[] expectedOutputs) {
		List<Double[][]> weightDerivatives = new ArrayList<>();
		Double networkCost = getNetworkCost(expectedOutputs);

		// for each layer propagate backwards
		for (int layer = numLayers - 1; layer >= 1; layer--) {
			Double[] actualOutputs = activations.get(layer);
			// get costs: actual - expected
			Double[] costs = getCosts(actualOutputs, expectedOutputs);

			// backProp to prev layer: update weights for this layer & compute expected activations for prev layer
			Double[] expectedOutputsPrev = new Double[numNeuronsOnLayers[layer - 1]];
			Arrays.fill(expectedOutputsPrev, 0.0);

			Double[][] derivativesForWeights = new Double[numNeuronsOnLayers[layer - 1]][numNeuronsOnLayers[layer]];

			for (int neuron = 0; neuron < numNeuronsOnLayers[layer]; neuron++) {
				Double derivativeWoActivationOnPrev = 2 * costs[neuron] * actualOutputs[neuron] * (1 - actualOutputs[neuron]);
				for (int i = 0; i < numNeuronsOnLayers[layer - 1]; i++) {
					derivativesForWeights[i][neuron] = derivativeWoActivationOnPrev * activations.get(layer - 1)[i];
					expectedOutputsPrev[i] += derivativeWoActivationOnPrev * weights.get(layer - 1)[i][neuron];
				}
			}
			expectedOutputs = expectedOutputsPrev;
			weightDerivatives.add(numLayers - layer - 1, derivativesForWeights);
		}
		Collections.reverse(weightDerivatives);
		return new Pair<>(weightDerivatives, networkCost);
	}

	private Double getNetworkCost(Double[] expectedOutputs) {
		Double networkCosts[] = getCosts(activations.get(numLayers - 1), expectedOutputs);
		Double result = 0.0;
		for (Double d : networkCosts) {
			result += d * d;
		}
		return result;
	}

	private Double[] getColumnI(int i, Double[][] matrix) {
		Double result[] = new Double[matrix.length];
		for (int j = 0; j < result.length; j++) {
			result[i] = matrix[j][i];
		}
		return result;
	}

	private Double[] getCosts(Double[] actualOutputs, Double[] expectedOutputs) {
		Double costs[] = new Double[actualOutputs.length];
		for (int i = 0; i < actualOutputs.length; i++) {
			costs[i] = actualOutputs[i] - expectedOutputs[i];
		}
		return costs;
	}

	/**
	 * Performs a forward propagation through the neu-net, given the input-layer's activations
	 */
	private void forwardPass(Double... activations) {
		this.activations = new ArrayList<>();
		for (int i = 1; i < numLayers; i++) {
			Double currActivations[] = new Double[numNeuronsOnLayers[i]];
			for (int j = 0; j < numNeuronsOnLayers[i]; j++) {
				currActivations[j] = sigmoid(aMulW(weights.get(i - 1), activations, i, j));
			}
			this.activations.add(activations);
			activations = currActivations;
		}
		this.activations.add(activations);
	}

	private Double aMulW(Double[][] weights, Double[] activations, int layerNumber, int neuronOnLayer) {
		Double sum = 0.0;
		for (int i = 0; i < numNeuronsOnLayers[layerNumber - 1]; i++) {
			sum += weights[i][neuronOnLayer] * activations[i];
		}
		return sum;
	}

	/**
	 * Implements the sigmoid function for the input
	 */
	private Double sigmoid(Double z) {
		return 1.0 / (1 + Math.exp(-z));
	}

	/**
	 * Initializes and returns the weight-matrix between layers fromLayer -> fromLayer+1
	 */
	private Double[][] initializeWeights(int fromLayer) {
		Random random = new Random();
		Double weights[][] = new Double[numNeuronsOnLayers[fromLayer]][numNeuronsOnLayers[fromLayer + 1]];
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = random.nextDouble() % 1;
			}
		}
		return weights;
	}

	/**
	 * Imports and returns a neu-net from the file given as parameter
	 */
	public static KoboldNet importFromFile(String file) {
		KoboldNet koboldNet = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(file));

			koboldNet = new KoboldNet();

			int state = 0; // 0-start 1-numLayersReaded 2-neuronCountsReaded 3-WeightsReaded
			String line;
			int[] numNeuronsOnLayersInt = new int[0];
			int currentWeightMatrix = 0;
			while ((line = reader.readLine()) != null) {
				if (!line.isEmpty() && line.charAt(0) != '#') {
					switch (state) {
						case 0:
							int numLayers = Integer.parseInt(line);
							koboldNet.setNumLayers(numLayers);
							state++;
							break;
						case 1:
							String[] numNeuronsOnLayers = line.split(" ");
							numNeuronsOnLayersInt = new int[numNeuronsOnLayers.length];
							for (int i = 0; i < numNeuronsOnLayers.length; i++) {
								numNeuronsOnLayersInt[i] = Integer.parseInt(numNeuronsOnLayers[i]);
							}
							koboldNet.setNumNeuronsOnLayers(numNeuronsOnLayersInt);
							state++;
							break;
						case 2:
							Double[][] weights = new Double[numNeuronsOnLayersInt[currentWeightMatrix]][numNeuronsOnLayersInt[currentWeightMatrix + 1]];
							for (int i = 0; i < weights.length; i++) {
								String[] weightsString = line.split(" ");
								for (int j = 0; j < weightsString.length; j++) {
									weights[i][j] = Double.parseDouble(weightsString[j]);
								}
								line = reader.readLine();
							}
							koboldNet.addWeightMatrix(weights);
							currentWeightMatrix++;
							break;
					}
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return koboldNet;
	}

	/**
	 * Exports the neu-net into the file given as parameter
	 */
	public void exportToFile(String file) {
		try {
			PrintWriter writer = new PrintWriter(file, "UTF-8");

			writer.print("#number of layers in koboldNet\n" + numLayers + "\n");

			writer.print("#number of neurons on each layer (including input and output layers)\n");
			for (int i = 0; i < numLayers; i++) {
				writer.print(numNeuronsOnLayers[i] + " ");
			}
			writer.print("\n\n");

			writer.print("#WEIGHT MATRICES\n");
			for (int i = 0; i < weights.size(); i++) {
				writer.print("#between layer" + i + " and layer" + (i + 1) + "\n");
				Double[][] weightMatrix = weights.get(i);
				for (Double[] aWeightMatrix : weightMatrix) {
					for (Double anAWeightMatrix : aWeightMatrix) {
						writer.print(anAWeightMatrix + " ");
					}
					writer.print("\n");
				}
			}
			writer.flush();
			writer.close();
			System.out.println("Exporting koboldNet successFull. File: " + file);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Sets the number of layers of this instance
	 */
	private void setNumLayers(int numLayers) {
		this.numLayers = numLayers;
	}

	/**
	 * Sets the array holding the numbers of neurons on the layers of this instance
	 */
	private void setNumNeuronsOnLayers(int[] numNeuronsOnLayers) {
		this.numNeuronsOnLayers = numNeuronsOnLayers;
	}

	/**
	 * Adds a weight-matrix to the list of the weight-matrices of this neu-net
	 *
	 * @param weightMatrix the weight-matrix to be added to the list of the weight-matrices
	 */
	private void addWeightMatrix(Double[][] weightMatrix) {
		weights.add(weightMatrix);
	}
}
