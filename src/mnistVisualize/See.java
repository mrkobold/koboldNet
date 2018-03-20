package mnistVisualize;

import javax.swing.*;
import java.awt.*;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * Class to visualize the digits only to check that we are reading the accordingly
 */
@SuppressWarnings({"InfiniteLoopStatement", "ResultOfMethodCallIgnored"})
public class See {

	public static void main(String[] args) throws Exception {
		JFrame frame = new JFrame();
		frame.setSize(500, 500);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		Graphics g = frame.getGraphics();

		String file = "/home/boldizsarkovacs/koboldNetInputs/train-images.idx3-ubyte";

		DataInputStream inputStream = new DataInputStream(new FileInputStream(new File(file)));
		byte[] b = new byte[16];
		inputStream.read(b, 0, 16);
/*
		while (true) {
			int image[][] = new int[28][28];
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					image[i][j] = inputStream.readUnsignedByte();
					int color = image[i][j];
					g.setColor(new Color(color, color, color));
					g.drawRect(j * 2, i * 2, 2, 2);
				}
			}
			g.clearRect(0, 0, 56, 56);
			try {
				Thread.currentThread().wait(5000);
			} catch (Exception ignore) {

			}
		}*/


		String labelFile = "/home/boldizsarkovacs/koboldNetInputs/train-labels.idx1-ubyte";
		DataInputStream inputStreamLabel = new DataInputStream(new FileInputStream(new File(labelFile)));
		inputStreamLabel.read(b, 0, 8);

	}
}
