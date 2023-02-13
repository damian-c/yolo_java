package yolo_java;


import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;


// Container for all the things detected in an image
public class Composition {

	private Mat image;
	public List<Thing> things = new ArrayList<>();

	public Composition() {
	}

	public Composition(Mat image) {
		this.image = image;
	}

	public void setImage(Mat image) {
		this.image = image;
	}

	// returns new composition with fewer overlapping boxes
	public Composition reduceOverlap(float confThreshold ) {
		float nmsThresh = 0.5f;
		Composition outputScene = new Composition(image);
		List<Rect2d> boxList = new ArrayList<>();
		List<Float> confList = new ArrayList<>();
		MatOfRect2d boxes = new MatOfRect2d();
		MatOfFloat confidences = new MatOfFloat();
		MatOfInt indices = new MatOfInt();

		for (Thing thing : things) {
			boxList.add(thing.box);
			confList.add(thing.confidence);
		}
		boxes.fromList(boxList);
		confidences.fromList(confList);

		// Use non-maximum suppression to reduce overlapping boxes
		Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
		if (!indices.empty()) {
			for (int index : indices.toArray()) {
				Thing thing = things.get(index);
				outputScene.things.add(thing);
			}
		}
		return outputScene;
	}

	// returns an image in matrix format, with boxes around around the detected things
	public Mat drawBoxes() {
		int font = Imgproc.FONT_HERSHEY_PLAIN;
		Mat outputImage = image;

		for (Thing thing : things) {
			Imgproc.rectangle(outputImage, thing.box.tl(), thing.box.br(), thing.color, 2);
			Point text_point = new Point(thing.box.x, thing.box.y - 5);
			Imgproc.putText(outputImage, thing.label, text_point, font, 1.5, thing.color, 1);
		}
		return outputImage;
	}

}
