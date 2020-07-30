package example;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
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
        String imagePath = new File(savePath + "/finger_pointing_to_words_02.jpg").getAbsolutePath();
        // convert source image to HSV
        Mat srcImage = loadImage(imagePath);
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(srcImage, hsvImage, Imgproc.COLOR_BGR2HSV);
        // mask hsv image
        Scalar scalarLower = new Scalar(0, 0.28*255, 0);
        Scalar scalarUpper = new Scalar(25, 0.68*255, 255);
        Mat maskedImage = new Mat();
        Core.inRange(hsvImage, scalarLower, scalarUpper, maskedImage);
        // save masked image
        saveImage(maskedImage, savePath + "/finger_detected_02.jpg");
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
