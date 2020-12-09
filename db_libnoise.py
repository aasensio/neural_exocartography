import numpy as np
import matplotlib.pyplot as pl
import exocartographer
import healpy as hp
import zarr
from tqdm import tqdm
from opensimplex import OpenSimplex
from enum import IntEnum
from mpi4py import MPI
from PIL import Image
import subprocess

import scipy.sparse.linalg
class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def simplex_noise(noise, x, y, z, freq, weight):
    return weight * (noise.noise3d(x*freq, y*freq, z*freq) / 2.0 + 0.5)

def billow_noise(noise, x, y, z, freq, weight):
    return weight * (2.0 * np.abs(noise.noise3d(x*freq, y*freq, z*freq)) - 1.0)

def conductor_matrix(size_world, n_batches, n_per_batch):
    n = n_batches * n_per_batch

    p_rotation_arr = np.random.uniform(low=10.0, high=50.0, size=(n_batches, n_per_batch))
    p_orbit_arr = np.random.uniform(low=100.0, high=500.0, size=(n_batches, n_per_batch))
    obliquity_arr = np.random.uniform(low=0.0, high=90.0, size=(n_batches, n_per_batch))
    inclination_arr = np.random.uniform(low=0.0, high=90.0, size=(n_batches, n_per_batch))

    sim_nside = 16
    npix = hp.nside2npix(sim_nside)

    nobs_per_epoch = 24

    handler = zarr.open('training_matrices.zarr', 'w')
    ds_pars = handler.create_dataset('pars', shape=(n, 6), chunk=(1, None), dtype='float32')
    ds_largest_eval = handler.create_dataset('largest_eval', shape=(n), dtype='float32')
    ds_matrix = handler.create_dataset('matrix', shape=(n, nobs_per_epoch * 5, npix), chunk=(1, None, None), dtype='float32')

    tasks = [i for i in range(n_batches)]

    task_index = 0
    num_workers = size_world - 1
    closed_workers = 0
    print(f"*** Master starting for computing {n} matrices")
    print("*** Master starting with {0} workers".format(num_workers))
    while closed_workers < num_workers:        
        dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):

                dataToSend = {'index': task_index, 'p_rotation': p_rotation_arr[task_index, :], 'p_orbit': p_orbit_arr[task_index, :], \
                    'obliquity': obliquity_arr[task_index, :], 'inclination': inclination_arr[task_index, :], 
                    'sim_nside': sim_nside, 'nobs_per_epoch': nobs_per_epoch, 'nx': nobs_per_epoch * 5, 'ny': npix}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0}/{1} to worker {2}".format(task_index, n_batches, source), flush=True)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:            
            index = dataReceived['index']
            matrix = dataReceived['matrix']
            pars = dataReceived['pars']
            largest_eval = dataReceived['eval']

            print(" * MASTER : got block {0} from worker {1}".format(index, source), flush=True)
            
            ds_matrix[n_per_batch*index : n_per_batch*(index+1), :, :] = matrix
            ds_pars[n_per_batch*index : n_per_batch*(index+1), :] = pars        
            ds_largest_eval[n_per_batch*index : n_per_batch*(index+1)] = largest_eval
                
            print(" * MASTER : got block {0} from worker {1} - saved".format(index, source), flush=True)
                
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master finishing")


def conductor_surface(size_world, n_batches, n_per_batch):
    n = n_batches * n_per_batch

    sim_nside = 16
    npix = hp.nside2npix(sim_nside)

    delta_lat = 180.0 / 512.0
    lat = 90.0 - delta_lat * np.arange(512)

    delta_lon = 360.0 / 1024.0
    lon = delta_lon * np.arange(1024)

    LAT, LON = np.meshgrid(lat, lon)
    
    ind = hp.ang2pix(sim_nside, LON, LAT, lonlat=True)
    
    handler = zarr.open('training_surfaces_libnoise.zarr', 'w')
    ds_surface = handler.create_dataset('surface', shape=(n, npix), chunk=(1, None), dtype='float32')

    tasks = [i for i in range(n_batches)]

    task_index = 0
    num_workers = size_world - 1
    closed_workers = 0
    print(f"*** Master starting for computing {n} surfaces")
    print("*** Master starting with {0} workers".format(num_workers))
    while closed_workers < num_workers:        
        dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):

                dataToSend = {'index': task_index, 'ind': ind, 'npix': npix, 'n_maps': n_per_batch}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0}/{1} to worker {2}".format(task_index, n_batches, source), flush=True)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:            
            index = dataReceived['index']
            surface = dataReceived['surface']
            
            for i in range(n_per_batch):                
                ds_surface[n_per_batch*index:n_per_batch*(index+1), :] = surface
                
            print(" * MASTER : got block {0} from worker {1} - saved".format(index, source), flush=True)
                
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master finishing")

def conductor_clouds(size_world, n_batches, n_per_batch):
    n = n_batches * n_per_batch

    sim_nside = 16
    npix = hp.nside2npix(sim_nside)

    polar_angle, azimuthal_angle = hp.pixelfunc.pix2ang(sim_nside, np.arange(npix))

    x = np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = np.cos(polar_angle)
    
    handler = zarr.open('training_clouds.zarr', 'w')
    ds_surface = handler.create_dataset('clouds', shape=(n, npix), chunk=(1, None), dtype='float32')

    tasks = [i for i in range(n_batches)]

    task_index = 0
    num_workers = size_world - 1
    closed_workers = 0
    print(f"*** Master starting for computing {n} surfaces")
    print("*** Master starting with {0} workers".format(num_workers))
    while closed_workers < num_workers:        
        dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):

                dataToSend = {'index': task_index, 'x': x, 'y': y, 'z': z, 'npix': npix, 'n_maps': n_per_batch}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0}/{1} to worker {2}".format(task_index, n_batches, source), flush=True)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:            
            index = dataReceived['index']
            surface = dataReceived['surface']
            
            for i in range(n_per_batch):                
                ds_surface[n_per_batch*index:n_per_batch*(index+1), :] = surface
                
            print(" * MASTER : got block {0} from worker {1} - saved".format(index, source), flush=True)
                
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master finishing")

def worker_matrix():

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            p_rotation_arr = dataReceived['p_rotation']
            p_orbit_arr = dataReceived['p_orbit']
            obliquity_arr = dataReceived['obliquity']
            inclination_arr = dataReceived['inclination']
            sim_nside = dataReceived['sim_nside']
            nobs_per_epoch = dataReceived['nobs_per_epoch']
            nx = dataReceived['nx']
            ny = dataReceived['ny']

            npix = hp.nside2npix(sim_nside)

            n = len(p_rotation_arr)

            matrix = np.zeros((n, nx, ny))
            pars = np.zeros((n, 6))
            largest_eval = np.zeros(n)

            for i in range(n):
            
                # Set orbital properties
                p_rotation = p_rotation_arr[i]    # hours
                p_orbit = p_orbit_arr[i] * 24.0   # hours
                
                phi_orb = np.pi
                inclination = inclination_arr[i] * np.pi/180.0
                obliquity = obliquity_arr[i] * np.pi/180.0
                phi_rot = np.pi

                # Observation schedule
                cadence = p_rotation/24.        
                epoch_duration = nobs_per_epoch * cadence                            

                epochs = np.random.uniform(low=0.0, high=1.0, size=5)        
                epoch_starts = p_orbit * np.sort(epochs)        
                epoch_starts = [30*p_rotation, 60*p_rotation, 150*p_rotation,
                                210*p_rotation, 250*p_rotation]

                times = np.array([])
                for epoch_start in epoch_starts:
                    epoch_times = np.linspace(epoch_start,
                                            epoch_start + epoch_duration,
                                            nobs_per_epoch)
                    times = np.concatenate([times, epoch_times])

                measurement_std = 0.001

                truth = exocartographer.IlluminationMapPosterior(times, np.zeros_like(times),
                                                measurement_std, nside=sim_nside, nside_illum=sim_nside)

                true_params = {
                    'log_orbital_period':np.log(p_orbit),
                    'log_rotation_period':np.log(p_rotation),
                    'logit_cos_inc':exocartographer.util.logit(np.cos(inclination)),
                    'logit_cos_obl':exocartographer.util.logit(np.cos(obliquity)),
                    'logit_phi_orb':exocartographer.util.logit(phi_orb, low=0, high=2*np.pi),
                    'logit_obl_orientation':exocartographer.util.logit(phi_rot, low=0, high=2*np.pi)}
                truth.fix_params(true_params)
                p = np.concatenate([np.zeros(truth.nparams), np.zeros(npix)])

                matrix[i, :, :] = truth.visibility_illumination_matrix(p)
                largest_eval[i] = scipy.sparse.linalg.eigsh(matrix[i, :, :].T @ matrix[i, :, :], k=1, which='LM', return_eigenvectors=False)                
                
                pars[i, :] = np.array([p_orbit, p_rotation, inclination, obliquity, phi_orb, phi_rot])

            dataToSend = {'index': task_index, 'matrix': matrix, 'pars': pars, 'eval': largest_eval}
            comm.send(dataToSend, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break        
    
    comm.send(None, dest=0, tag=tags.EXIT)


def worker_surface():

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            ind = dataReceived['ind']
            n = dataReceived['n_maps']
            npix = dataReceived['npix']
                        
            surface = np.zeros((n, npix))
            
            for j in range(n):

                seed = np.random.randint(low=0, high=1000000)
                process = subprocess.run(["./noise", "simple", f"{seed}"])
                
                simulated_map = np.zeros(npix)
                planet = np.array(Image.open(f'planet-{seed}-specular.bmp'))[:, :, 0]

                surface[j, ind.T] = planet / 255.0

                process = subprocess.run(["rm", "-f", f"planet-{seed}-specular.bmp"])
                process = subprocess.run(["rm", "-f", f"planet-{seed}-surface.bmp"])
                process = subprocess.run(["rm", "-f", f"planet-{seed}-normal.bmp"])

            dataToSend = {'index': task_index, 'surface': surface}
            comm.send(dataToSend, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break        
    
    comm.send(None, dest=0, tag=tags.EXIT)

def worker_clouds():

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            x = dataReceived['x']
            y = dataReceived['y']
            z = dataReceived['z']
            n = dataReceived['n_maps']
            npix = dataReceived['npix']
                        
            surface = np.zeros((n, npix))
            
            for k in range(n):

                seed = np.random.randint(low=0, high=100000)
                n_octaves = 5
                noises = [None] * n_octaves
                for i in range(n_octaves):        
                    noises[i] = OpenSimplex(seed=seed+i)

                simulated_map = np.zeros(npix)
    
                for i in range(npix):
                    freq = 1.0
                    persistence = 1.0
                    for j in range(5):
                        simulated_map[i] += billow_noise(noises[j], x[i], y[i], z[i], freq, persistence)
                        freq *= 1.65
                        persistence *= 0.5

                thr = np.random.uniform(low=0.2, high=0.7, size=1)[0]
                mx = np.max(simulated_map)
                mn = np.min(simulated_map)
                simulated_map = (simulated_map - mn) / (mx - mn)
                simulated_map[simulated_map < thr] = 0.0

                mx = np.max(simulated_map[simulated_map > thr])
                mn = np.min(simulated_map[simulated_map > thr])
                simulated_map[simulated_map > thr] = (simulated_map[simulated_map > thr] - mn) / (mx - mn)
                
                surface[k, :] = simulated_map

            dataToSend = {'index': task_index, 'surface': surface}
            comm.send(dataToSend, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break        
    
    comm.send(None, dest=0, tag=tags.EXIT)
        
if (__name__ == '__main__'):
    
# Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    n_batches = 100
    n_per_batch = 50

    if (rank == 0):

        conductor_matrix(size, n_batches, n_per_batch)
        
    else:
        worker_matrix()

    comm.Barrier()
    comm.Barrier()

    n_batches = 100
    n_per_batch = 50

    if (rank == 0):

        conductor_surface(size, n_batches, n_per_batch)
        
    else:
        worker_surface()

    n_batches = 100
    n_per_batch = 50

    if (rank == 0):

        conductor_clouds(size, n_batches, n_per_batch)
        
    else:
        worker_clouds()