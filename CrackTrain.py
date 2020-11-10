from model import *
from data import *
import image_proc as ip
import signal_proc as sp
import geo_proc as gp
import os
import matplotlib
import glob
import geopandas


# continues training the model with weights in old_weight_path. If None, starts from scratch
# and saves it to working directory.
# New weights will be saved to a file in new_weight_path. If None, it will instead overwrite the old one.
def train_main(work_dir=None, old_weight_path=None,
               new_weight_path=None):
    orig_img_dir = work_dir + '/Training/Images/Originals'
    shp_bounds_dir = work_dir + '/Training/Shapefiles/Areas'
    shp_dir = work_dir + '/Training/Shapefiles/Labels'
    gen_dir = work_dir + '/Training/Images/Generated'
    if new_weight_path is None:
        new_weight_path = work_dir + '/unet_weights.hdf5'
    model = unet(old_weight_path)

    def filebrowser(dir, ext=''):
        return [f for f in glob.glob(f'{dir}/*{ext}')]

    img_names = filebrowser(orig_img_dir, '.png')
    shp_names = filebrowser(shp_dir, '.shp')
    bound_names = filebrowser(shp_bounds_dir, '.shp')
    training_list = list()
    for i, file in enumerate(img_names):
        area_name = file[(len(orig_img_dir) + 1):(len(file) - 4)]
        for j, shp in enumerate(shp_names):
            shp_name = shp[(len(shp_dir) + 1):(len(shp) - 4)]
            if area_name in shp_name:
                for k, ar in enumerate(bound_names):
                    ar_name = ar[(len(shp_bounds_dir) + 1):(
                                len(ar) - 4)]
                    if area_name in ar_name:
                        training_list.append((img_names[i],
                                              shp_names[j],
                                              bound_names[k]))
                        break
                else:
                    print(
                        f'Could not find a .shp file in {shp_bounds_dir} representing the area in image {file} and .shp file {shp}')
                break
        else:
            print(
                f'Could not find a .shp file representing image {file}')

    assert len(
        training_list) > 0, 'Could not find any .png-.shp file pairs.'

    for ind, (img, geo_file, geo_bounds) in enumerate(
            training_list):
        img = ip.open_image(img)
        geo_data = geopandas.GeoDataFrame.from_file(geo_file)
        geo_area = geopandas.GeoDataFrame.from_file(geo_bounds)
        label_bin_img = gp.geo_dataframe_to_binmat(geo_data,
                                                   img.shape,
                                                   relative_geo_data=geo_area,
                                                   slack=0)

        # divide original img (and binary images) into sub images of shape (h, w) and save them
        w, h = (256, 256)
        sub_imgs = ip.img_segmentation(img, width=w, height=h)
        sub_bin_imgs = ip.img_segmentation(label_bin_img, width=w,
                                           height=h)
        suf = 1
        for (im, b) in zip(sub_imgs, sub_bin_imgs):
            if np.quantile(im, 0.95) == 0 or np.quantile(im,
                                                         0.05) == 255:
                continue
            ip.save_image(
                gen_dir + '/Source/src' + str(ind) + '_' + str(
                    suf) + '.png', im)
            ip.save_image(
                gen_dir + '/Labels/lbl' + str(ind) + '_' + str(
                    suf) + '.png', b)
            suf += 1

    # set up the training generator's image altering parameters
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    train_generator = trainGenerator(4, gen_dir, 'Source', 'Labels',
                                     data_gen_args,
                                     save_to_dir=None)

    model_checkpoint = ModelCheckpoint(new_weight_path,
                                       monitor='loss', verbose=1,
                                       save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=300,
                        epochs=5, callbacks=[model_checkpoint])

    pass


# Tries to find necessary files for the training. If fails, it will create them to the working directory
def train_setup(work_dir=None):
    training_dirs = ['/Training', '/Training/Images',
                     '/Training/Shapefiles',
                     '/Training/Shapefiles/Areas',
                     '/Training/Images/Originals',
                     '/Training/Shapefiles/Labels',
                     '/Training/Images/Generated',
                     '/Training/Images/Generated/Source',
                     '/Training/Images/Generated/Predictions',
                     '/Training/Images/Generated/Labels']
    for dir in training_dirs:
        if dir == '/Training/Images/Generated':
            try:
                import shutil
                shutil.rmtree(work_dir + dir)
            except FileNotFoundError:
                pass
        try:
            os.makedirs(work_dir + dir)
        except OSError:
            pass

    pass


if __name__ == '__main__':
    work_dir = os.getcwd()
    train_setup(work_dir=work_dir)
    train_main(work_dir=work_dir)
