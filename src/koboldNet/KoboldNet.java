package koboldNet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * KoboldNet
 */
public class KoboldNet {

	List<Double[][]> weights = new ArrayList<>();

	// 1 input, 1 output, n-2 hidden layers
	int numLayers = 3;
	int numNeuronsOnLayers[] = new int[]{3, 4, 2};

	public void initialize() {
		for (int i = 0; i < numLayers - 1; i++) {
			Double[][] weightsI = initializeWeights(i);
			weights.add(i, weightsI);
		}
	}

	public void forwardPass(Double... activations) {
		for (int i = 1; i < numLayers; i++) {
			Double currActivations[] = new Double[numNeuronsOnLayers[i]];
			for (int j = 0; j < numNeuronsOnLayers[i]; j++) {
				currActivations[j] = sigmoid(aMulW(weights.get(i - 1), activations, i, j));
			}
			activations = currActivations;
		}

		System.out.println("Activations on last layer:");
		for (Double activation : activations) {
			System.out.println(activation + ",");
		}
	}

	private Double aMulW(Double[][] weights, Double[] activations, int layerNumber, int neuronOnLayer) {
		Double sum = 0.0;
		for (int i = 0; i < numNeuronsOnLayers[layerNumber - 1]; i++) {
			sum += weights[i][neuronOnLayer] * activations[i];
		}
		return sum;
	}

	private Double sigmoid(Double z) {
		return 1.0 / (1 + Math.exp(z));
	}

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

	public void setNumLayers(int numLayers) {
		this.numLayers = numLayers;
	}

	public void setNumNeuronsOnLayers(int[] numNeuronsOnLayers) {
		this.numNeuronsOnLayers = numNeuronsOnLayers;
	}

	public void addWeightMatrix(Double[][] weightMatrix) {
		weights.add(weightMatrix);
	}
}
