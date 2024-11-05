from gradcam.feature_extraction.CAMs import CAM
from gradcam.utils.visualization import CAMFeatureMaps

feature_maps = CAMFeatureMaps(CAM)
extracting_model =1
extracting_module=1
targeting_layer=1
feature_maps.load(extracting_model,extracting_module,targeting_layer,has_gap='0')
mask = feature_maps.show(X_test[0], None)
feature_maps.map_activation_to_input(mask)