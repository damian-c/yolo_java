package yolo_java;


import org.opencv.core.*;
import org.opencv.dnn.*;

import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.util.Scanner;
import java.io.FileNotFoundException;


public class Vision {

	private List<String> labels = new ArrayList<String>();
	private Net net;
	private float confThreshold = 0.5f;

	public Vision(String configFile, String weightsFile, String namesFile, float confThreshold) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load the openCV 4.x dll
		net = Dnn.readNetFromDarknet(configFile, weightsFile); // OpenCV DNN supports models trained from various frameworks like Caffe and TensorFlow. It also supports various networks architectures based on YOLO
		try { // Read list of label names from file
			File f = new File(namesFile);
			Scanner reader = new Scanner(f);
			while (reader.hasNextLine()) {
				String class_name = reader.nextLine();
				labels.add(class_name);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();	
		}
		this.confThreshold = confThreshold;
	}

	public Composition detectThings(Mat image) {
		Composition scene = new Composition(image);
		Size sz = new Size(288,288);	    
		List<Mat> results = new ArrayList<>();
		List<String> outBlobNames = getOutputNames();
		Mat blob = Dnn.blobFromImage(image, 0.00392, sz, new Scalar(0), true, false); // We feed one frame of video into the network at a time, we have to convert the image to a blob. A blob is a pre-processed image that serves as the input.
		net.setInput(blob);
		net.forward(results, outBlobNames); // Feed forward the model to get output

		for (Mat level : results) {
			// each row is a candidate detection, the 1st 4 numbers are
			// [center_x, center_y, width, height], followed by (N-4) class probabilities
			for (int j = 0; j < level.rows(); ++j) {
				Mat row = level.row(j);
				Mat scores = row.colRange(5, level.cols());
				Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
				float confidence = (float)mm.maxVal;
				Point classIdPoint = mm.maxLoc;
				if (confidence > confThreshold) {
					int centerX = (int)(row.get(0,0)[0] * image.cols()); // scaling for drawing the bounding boxes
					int centerY = (int)(row.get(0,1)[0] * image.rows());
					int width   = (int)(row.get(0,2)[0] * image.cols());
					int height  = (int)(row.get(0,3)[0] * image.rows());
					int left    = centerX - width  / 2;
					int top     = centerY - height / 2;
					String label = labels.get((int)classIdPoint.x);
					Thing thing = new Thing(label, confidence, new Rect2d(left, top, width, height));
					scene.things.add(thing);
				}
			}
		}
		return scene;
	}

	private List<String> getOutputNames() {
		List<String> names = new ArrayList<>();
		List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
		List<String> layersNames = net.getLayerNames();
		outLayers.forEach((item) -> names.add(layersNames.get(item - 1))); // unfold and create R-CNN layers from the loaded YOLO model
		return names;
	}

}