from __imp__ import *

def initGraph(frozengraph):
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(frozengraph, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
	return detection_graph,category_index

def run_inference_for_single_image(image, graph,tensor_dict,sess):
	if 'detection_masks' in tensor_dict:
		detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
		detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

		real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
		detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
		detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
			detection_masks, detection_boxes, image.shape[0], image.shape[1])
		detection_masks_reframed = tf.cast(
			tf.greater(detection_masks_reframed, 0.5), tf.uint8)

		tensor_dict['detection_masks'] = tf.expand_dims(
			detection_masks_reframed, 0)
	image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

	output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

	output_dict['num_detections'] = int(output_dict['num_detections'][0])
	output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
	output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	output_dict['detection_scores'] = output_dict['detection_scores'][0]
	if 'detection_masks' in output_dict:
		output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict


def video(dgraph,cindex,media):
	cap = cv2.VideoCapture(media)

	try:
		with dgraph.as_default():
			with tf.Session() as sess:
					ops = tf.get_default_graph().get_operations()
					all_tensor_names = {output.name for op in ops for output in op.outputs}
					tensor_dict = {}
					for key in [
						'num_detections', 'detection_boxes', 'detection_scores',
						'detection_classes', 'detection_masks'
					]:
						tensor_name = key + ':0'
						if tensor_name in all_tensor_names:
							tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
							tensor_name)

					while True:
						ret, image_np = cap.read()
						image_np_expanded = np.expand_dims(image_np, axis=0)
						image_np=cv2.resize( image_np,(300,300) )
						output_dict = run_inference_for_single_image(image_np, dgraph, tensor_dict, sess)
						vis_util.visualize_boxes_and_labels_on_image_array(
							image_np,
							output_dict['detection_boxes'],
							output_dict['detection_classes'],
							output_dict['detection_scores'],
							cindex,
							instance_masks=output_dict.get('detection_masks'),
							use_normalized_coordinates=True,
							line_thickness=8,
							min_score_thresh=.85)
						cv2.imshow('object_detection', cv2.resize(image_np, (300, 300)))
						if cv2.waitKey(25) & 0xFF == ord('q'):
							cap.release()
							cv2.destroyAllWindows()
							break
	except Exception as e:
		print(e)
		cap.release()

def main():
	if len(sys.argv)!=3:
		print('Incorrect number of arguments')
		return
	if not os.path.exists( sys.argv[1] ):
		raise FileNotFoundError
	if not os.path.exists( sys.argv[2] ):
		raise FileNotFoundError

	dgraph,cindex=initGraph(sys.argv[2])
	video(dgraph,cindex,media=sys.argv[1])

if __name__ == "__main__":
	main()