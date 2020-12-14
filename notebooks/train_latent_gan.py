import numpy as np
import os.path as osp
# import matplotlib.pylab as plt

from src.point_net_ae import PointNetAutoEncoder
from src.autoencoder import Configuration as Conf
from src.neural_net import MODEL_SAVER_ID

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

# from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph

from src.vanilla_gan import Vanilla_GAN
from src.w_gan_gp import W_GAN_GP
from src.generators_discriminators import latent_code_discriminator_two_layers,\
latent_code_generator_two_layers



# Top-dir of where point-clouds are stored.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/'    

ae_configuration = '../data/single_class_ae/configuration'


# Where to save GANs check-points etc.
top_out_dir = '../data/'
experiment_name = 'latent_gan_with_chamfer_ae'

ae_epoch = 500           # Epoch of AE to load.
bneck_size = 128         # Bottleneck-size of the AE
n_pc_points = 2048       # Number of points per model.

class_name = "chair"


# Load point-clouds.
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
print ('Shape of DATA =', all_pc_data.point_clouds.shape)


# Load pre-trained AE
reset_tf_graph()
ae_conf = Conf.load(ae_configuration)
ae_conf.encoder_args['verbose'] = False
ae_conf.decoder_args['verbose'] = False
ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)
ae.restore_model(ae_conf.train_dir, ae_epoch, verbose=True)



# Use AE to convert raw pointclouds to latent codes.
latent_codes = ae.get_latent_codes(all_pc_data.point_clouds)
latent_data = PointCloudDataSet(latent_codes)
print ('Shape of DATA =', latent_data.point_clouds.shape)




# Check the decoded AE latent-codes look descent.
L = ae.decode(latent_codes)
i = 0
print(L[i])
i = 20
print(L[i])



# Set GAN parameters.

use_wgan = False     # Wasserstein with gradient penalty, or not?
n_epochs = 100        # Epochs to train.

plot_train_curve = True
save_gan_model = False
saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 50)])

# If true, every 'saver_step' epochs we produce & save synthetic pointclouds.
save_synthetic_samples = True
# How many synthetic samples to produce at each save step.
n_syn_samples = latent_data.num_examples

# Optimization parameters
init_lr = 0.0001
batch_size = 50
noise_params = {'mu':0, 'sigma': 0.2}
noise_dim = bneck_size
beta = 0.5 # ADAM's momentum.

n_out = [bneck_size] # Dimensionality of generated samples.

if save_synthetic_samples:
    synthetic_data_out_dir = osp.join(top_out_dir, 'OUT/synthetic_samples/', experiment_name)
    create_dir(synthetic_data_out_dir)

if save_gan_model:
    train_dir = osp.join(top_out_dir, 'OUT/latent_gan', experiment_name)
    create_dir(train_dir)
    
    
    
    
reset_tf_graph()

if use_wgan:
    lam = 10 # lambda of W-GAN-GP
    gan = W_GAN_GP(experiment_name, init_lr, lam, n_out, noise_dim, \
                  latent_code_discriminator_two_layers, 
                  latent_code_generator_two_layers,\
                  beta=beta)
else:    
    gan = Vanilla_GAN(experiment_name, init_lr, n_out, noise_dim,
                     latent_code_discriminator_two_layers, latent_code_generator_two_layers,
                     beta=beta)
    
    
    
    
accum_syn_data = []
train_stats = []




# Train the GAN.
for _ in range(n_epochs):
    loss, duration = gan._single_epoch_train(latent_data, batch_size, noise_params)
    epoch = int(gan.sess.run(gan.increment_epoch))
    print (epoch, loss)

    if save_gan_model and epoch in saver_step:
        checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
        gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

    if save_synthetic_samples and epoch in saver_step:
        syn_latent_data = gan.generate(n_syn_samples, noise_params)
        syn_data = ae.decode(syn_latent_data)
        np.savez(osp.join(synthetic_data_out_dir, 'epoch_' + str(epoch)), syn_data)
        for k in range(3):  # plot three (syntetic) random examples.
            print(syn_data[k][:, 0], syn_data[k][:, 1], syn_data[k][:, 2])

    train_stats.append((epoch, ) + loss)
    
    
    
    
    
