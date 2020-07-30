package example;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.net.URL;

public class OpencvExample {

    public static void main(String[] args) {
        // load opencv library
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
//        System.out.println("mat = " + mat.dump());

        // finger detection with hsv
        String savePath = "src/main/resources";
        String imagePath = new File(savePath + "/hand.jpg").getAbsolutePath();
        // convert source image to HSV
        Mat srcImage = loadImage(imagePath);
        Mat blurImage = new Mat();
        Mat hsvImage = new Mat();
        Imgproc.blur(srcImage, blurImage, new Size(7, 7));  // remove some noise
        Imgproc.cvtColor(blurImage, hsvImage, Imgproc.COLOR_BGR2HSV);
        // mask hsv image
        Scalar scalarLower = new Scalar(0, 0.28*255, 0);
        Scalar scalarUpper = new Scalar(25, 0.68*255, 255);
        Mat maskedImage = new Mat();
        Core.inRange(hsvImage, scalarLower, scalarUpper, maskedImage);
        // save masked image
        saveImage(maskedImage, savePath + "/hand_detected.jpg");
    }

    // load image
    public static Mat loadImage(String imagePath) {
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(imagePath);
    }
    // save image
    public static void saveImage(Mat imageMatrix, String targetPath) {
        Imgcodecs imgcodecs = new Imgcodecs();
        imgcodecs.imwrite(targetPath, imageMatrix);
    }

}
