package koboldNet;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

/**
 * Class containing methods for importing and exporting a neural network
 */
@SuppressWarnings("unused")
public class IOUtil {

	public static void readBatch(List<Double[]> inputs, List<Double[]> labels, String imagesFile, String labelsFile) {
		try {
			DataInputStream imagesStream = new DataInputStream(new FileInputStream(new File(imagesFile)));
			DataInputStream labelsStream = new DataInputStream(new FileInputStream(new File(labelsFile)));
			for (int i = 0; i < 100; i++) {
				Double image[] = new Double[28 * 28];
				Double label[] = new Double[10];

				for (int j = 0; j < 28; j++) {
					for (int k = 0; k < 28; k++) {
						image[28 * j + k] = (double) imagesStream.readUnsignedByte();
					}
				}

				Arrays.fill(label, 0.0);
				label[labelsStream.readUnsignedByte()] = 1.0;

				inputs.add(image);
				labels.add(label);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Imports and returns a neu-net from the file given as parameter
	 */
	public static KoboldNet importNetFromFile(String file) {
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
	public static void exportNetToFile(String file, KoboldNet koboldNet) {
		try {
			PrintWriter writer = new PrintWriter(file, "UTF-8");

			writer.print("#number of layers in koboldNet\n" + koboldNet.getNumLayers() + "\n");

			writer.print("#number of neurons on each layer (including input and output layers)\n");
			for (int i = 0; i < koboldNet.getNumLayers(); i++) {
				writer.print(koboldNet.getNumNeuronsOnLayers()[i] + " ");
			}
			writer.print("\n\n");

			writer.print("#WEIGHT MATRICES\n");
			for (int i = 0; i < koboldNet.getWeights().size(); i++) {
				writer.print("#between layer" + i + " and layer" + (i + 1) + "\n");
				Double[][] weightMatrix = koboldNet.getWeights().get(i);
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
}
