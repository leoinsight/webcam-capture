package example;

import ch.qos.logback.core.net.SyslogOutputStream;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32FC2;

public class OpencvExample {

    public static void main(String[] args) {
        // load opencv library
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
//        System.out.println("mat = " + mat.dump());

        // finger detection with hsv
        String savePath = "src/main/resources";
        String imagePath = new File(savePath + "/gongcha_menu.png").getAbsolutePath();
        // convert source image to HSV
        Mat srcImage = loadImage(imagePath);
        Mat blurImage = new Mat();
        Mat hsvImage = new Mat();
        Imgproc.blur(srcImage, blurImage, new Size(7, 7));  // remove some noise
        Imgproc.cvtColor(blurImage, hsvImage, Imgproc.COLOR_BGR2HSV);
        // mask hsv image
        Scalar scalarLower = new Scalar(0, 30, 0);
        Scalar scalarUpper = new Scalar(15, 255, 255);
        Mat maskedImage = new Mat();
        Core.inRange(hsvImage, scalarLower, scalarUpper, maskedImage);
        // save masked image
        saveImage(maskedImage, savePath + "/gongcha_menu_detected.png");

        // morphological operators
        // dilate with large element, erode with small ones
        Mat morphImage = new Mat();
        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        // draw contours
        Mat contImage = srcImage.clone();
        Imgproc.drawContours(contImage, contours, -1, new Scalar(0, 255, 0), 2);
        saveImage(contImage, savePath + "/gongcha_menu_contour.png");

        int centerX = srcImage.width() / 2;
        int bottom = srcImage.height() - 1;
        MatOfPoint maxContour = new MatOfPoint();
//        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint cont = contours.get(0);
            Point[] pointArr = cont.toArray();
            for (int j = 0; j < pointArr.length; j++) {
                Imgproc.drawMarker(contImage, pointArr[j], new Scalar(0, 255, 0), 1);
                // center x of src image is between min and max x value of contour?
//                if ()

                System.out.println("contour point:" + pointArr[j].x + "," + pointArr[j].y);
            }
//        }

        // determines whether the point is inside a contour
        Point pt = new Point(centerX, bottom-1);
        MatOfPoint2f con2 = new MatOfPoint2f();
        cont.convertTo(con2, CV_32FC2);
        double pointInContour = Imgproc.pointPolygonTest(con2, pt, false);
        System.out.println("src image center x:" + centerX);
        System.out.println("src image max y-1:" + bottom);
        System.out.println("point is inside?:" + pointInContour);

        // draw convex hull
        MatOfPoint points = contours.get(0);
        List<MatOfPoint> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(points, hull);
            Point[] contourArray = points.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }
        Mat convImage = contImage.clone();
        Imgproc.drawContours(convImage, hullList, -1, new Scalar(0, 0, 255), 2);
        saveImage(convImage, savePath + "/gongcha_menu_convex_hull.png");
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

    // find max contour
    public static MatOfPoint findMaxContour(List<MatOfPoint> contours) {
        MatOfPoint maxContour = new MatOfPoint();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);

            Rect rect = Imgproc.boundingRect(contour);
            if(rect.width > rect.height) continue;

            if(i == 0) maxContour = contour;
            double contourArea = Imgproc.contourArea(contour);
            double maxContourArea = Imgproc.contourArea(maxContour);
            if(contourArea > maxContourArea) maxContour = contour;
        }
        return maxContour;
    }
}
