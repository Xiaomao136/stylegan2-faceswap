import numpy as np
import dnnlib
import projector
import pretrained_networks
from training import dataset
from training import misc
import os


def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1, 1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1, 1])
            # print('###step', proj.get_cur_step())
            # dlatents = proj.get_dlatents()
            # for dlatents1 in dlatents:
            #     for dlatents2 in dlatents1:
            #         str = ''
            #         for e in dlatents2:
            #             str = '{} {}'.format(str, e)
            #         print('###', str)
    print('\r%-30s\r' % '', end='', flush=True)


def project_real_images(Gs, data_dir, dataset_name, snapshot_name, seq_no, num_snapshots=5):
    proj = projector.Projector()
    proj.set_network(Gs)

    print('Loading images from "%s/%s"...' % (data_dir, dataset_name))
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False,
                                       shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]


    print('Projecting image ...')
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('%s/image%04d-' % (snapshot_name, seq_no)),
                  num_snapshots=num_snapshots)
    ####################
    # print dlatents
    ####################
    dlatents = proj.get_dlatents()
    # for dlatents1 in dlatents:
    #     for dlatents2 in dlatents1:
    #         str = ''
    #         for e in dlatents2:
    #             str = '{} {}'.format(str, e)
    #         print('###', str)
    # img_name = f'100-100_01'
    # dir = 'results/dst'
    # img_name = '100-100_01.npy'
    # dir = 'results/src'
    # img_name = 'me_01.npy'
    # np.save(os.path.join(dir, img_name), dlatents[0])
    return dlatents[0]
