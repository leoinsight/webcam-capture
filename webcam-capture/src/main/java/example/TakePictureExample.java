package example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;

import javax.imageio.ImageIO;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.core.HttpHeaders;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.sarxos.webcam.Webcam;
import org.apache.commons.io.FileUtils;


/**
 * Example of how to take single picture.
 * 
 * @author Bartosz Firyn (SarXos)
 */
public class TakePictureExample {

	public static void main(String[] args) throws IOException {

//		// get default webcam and open it
//		Webcam webcam = Webcam.getDefault();
//		webcam.open();
//
//		// get image
//		BufferedImage image = webcam.getImage();
//
//		// save image to PNG file
//		ImageIO.write(image, "PNG", new File("test.png"));


		TakePictureExample main = new TakePictureExample();
		File file = main.getFileFromResources("HelloWorld.png");
		System.out.println("Working Directory = " + file);

		// read and convert image file to base64 string
		byte[] fileContent = FileUtils.readFileToByteArray(file);
		String encodedString = Base64.getEncoder().encodeToString(fileContent);
		System.out.println("Encoded String = " + encodedString);

		// prepare for Naver OCR API call
		final String OCR_KEY = "RGVQck1qYm1IZ29GZHJQWmpIdERNZlFqdFl6dHdrYXM=";
		final String REST_URI = "https://569cdd509a4c4e31bf80651af98c4b45.apigw.ntruss.com/custom/v1/2871/015a8ef25eb2f464f0bc251746946304749b818b0dda137b9041e56e4a965dbd/general";
		OcrReqImage ocrReqImg = new OcrReqImage("png", "HelloWorld");
		ocrReqImg.setData(encodedString);
		ArrayList<OcrReqImage> images = new ArrayList<>();
		images.add(ocrReqImg);
		OcrRequest ocrReq = new OcrRequest("V2", "string", 0, "ko");
		ocrReq.setImages(images);
		// call Naver OCR API
		String ocrApiCallResult = main.callOcrApi(REST_URI, ocrReq, OCR_KEY);
		System.out.println("OCR API Call Result = " + ocrApiCallResult);

		// extract OCR text
		String ocrText = main.extractOcrText(ocrApiCallResult, OcrResponse.class);
		System.out.println("Extracted OCR Text = " + ocrText);
	}

	// get file from classpath, resources folder
	private File getFileFromResources(String fileName) {
		ClassLoader classLoader = getClass().getClassLoader();
		URL resource = classLoader.getResource(fileName);
		if (resource == null) {
			throw new IllegalArgumentException("file is not found!");
		} else {
			return new File(resource.getFile());
		}
	}

	// call Naver OCR API
	private String callOcrApi(String REST_URI, OcrRequest ocrReq, String OCR_KEY) {
		Client client = ClientBuilder.newClient();
		Response res = client.target(REST_URI)
				.request(MediaType.APPLICATION_JSON)
				.header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON)
				.header("X-OCR-SECRET", OCR_KEY)
				.post(Entity.entity(ocrReq, MediaType.APPLICATION_JSON));
		return res.readEntity(String.class);
	}

	// extract OCR text
	private String extractOcrText(String ocrApiCallResult, Class<OcrResponse> ocrResponseClass) throws JsonProcessingException {
		ObjectMapper mapper = new ObjectMapper();
		OcrResponse ocrRes = mapper.readValue(ocrApiCallResult, ocrResponseClass);
		ArrayList<OcrResImage> ocrResImages = ocrRes.getImages();
		ArrayList<String> ocrTextList = new ArrayList<>();
		for (OcrResImage ocrResImage : ocrResImages) {
			ArrayList<OcrResField> ocrResFields = ocrResImage.getFields();
			for (OcrResField ocrResField : ocrResFields) {
				ocrTextList.add(ocrResField.getInferText());
			}
		}
		return String.join(" ", ocrTextList);
	}
}
