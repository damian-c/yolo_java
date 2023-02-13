package yolo_java;


import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;


// Represents an object in a scene
public class Thing {
	
	public final String label;
	public final float confidence;
	public final Rect2d box;
	public final Scalar color; // Scalar color format: (B,G,R)

	public Thing(String label, float confidence, Rect2d box) {
		this.label = label;
		this.confidence = confidence;
		this.box = box;
		this.color = new Scalar(0,0,255); // default color: red
	}

}
