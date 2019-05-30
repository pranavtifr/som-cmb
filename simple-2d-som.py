#! /usr/bin/env python
"""Use SOM for dimensionality reduction."""
import minisom 
import healpy as hp
import numpy as np
import logging
import sklearn.cluster as clus
import somoclu


def map_cleanup(mapps, in_mask=False):
    """
    Seperate the masked and unmaked region.

    Parameters
    ----------
    Input Map: Numpy Array

    Returns
    -------
    Cleaned Map, Masked Map, Unmasked Indices, Masked Indices : Numpy Array

    Raises
    ------
    None

    See Also
    --------
    None

    Notes
    -----
    None

    """
    if not in_mask:
        in_mask = np.ones(len(mapps))
    else:
        in_mask = hp.read_map("./mask.fits")
    cleaned_mapps = []
    masked_mapps = []
    outlier_indices = []
    map_indices = []
    for ite in range(len(mapps)):
        mask = in_mask[ite]
        if mask:
            cleaned_mapps.append(mapps[ite])
            map_indices.append(ite)
        else:
            masked_mapps.append(mapps[ite])
            outlier_indices.append(ite)
    return np.array(cleaned_mapps), np.array(masked_mapps), outlier_indices, map_indices


def read_maps(res):
    """
    Read all the freqency maps and make it into a numpy array.

    Parameters
    ----------
    resolution: Int

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    None

    Notes
    -----
    None

    """

    if res == 45:
        lmax = 4000
    elif res == 60:
        lmax = 3000
    else:
        logging.info("Invalid Value of res")
        raise ValueError
    source = "/user1/pranav/msc_codes/cmb-maps/data-nocmbsz/"
    maps = []
    for y in range(1,13):
        filename = f"/{res}/HFI_SimMap_y{y}_2048_R1.10_nominal_rebeam{res}lmax{lmax}.fits"
        mapp = hp.read_map(source+filename, verbose=False)
        logging.info(f"For y = {y} : Min:{np.min(mapp)} Max:{np.max(mapp)}")
        maps.append(mapp)
    return np.array(maps).T


def som(maps, kk=13):
    """
    SOM Using minisom library

    Parameters
    ----------
    Maps: Numpy Array

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    somoclu_cluster()
    Notes
    -----
    Clusters using kmeans clustering after dimensionality reduction.
    Writes the clusters to a fits file.
    """

    dim = 50 # int(np.sqrt(len(maps)))
    som = minisom.MiniSom(dim, dim, 12, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
    som.train_random(maps, 100) # trains the SOM with 100 iterations
    listwinners = []
    for x in maps:
        winners = som.winner(x)
        listwinners.append(winners)
    listwinners = np.array(listwinners)
    np.savetxt("listwinners",listwinners)
    for i in range(10):
        logging.info(f"Clustering with State {i}")
        clustermethod = clus.MiniBatchKMeans(n_clusters=kk, n_init = 10, random_state=i)
        pred = clustermethod.fit_predict(listwinners)
        inertia = clustermethod.inertia_
        logging.info(set(pred))
        hp.write_map(f"som-cluster_{i}.fits", pred)


def somoclu_cluster(maps, kk=13):
    """
    SOM Using somoclu library

    Parameters
    ----------
    Maps: Numpy Array

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    som()
    Notes
    -----
    Clusters using kmeans clustering after dimensionality reduction.
    Writes the clusters to a fits file.
    """
    logging.info(f"Preprocessing..")
    from sklearn import preprocessing
    clean_map, masked_map,masked, notmasked = map_cleanup(maps,True)
    
    scaler = preprocessing.RobustScaler(quantile_range=(15., 85.))
    clean_map = scaler.fit_transform(clean_map)
    masked_map = scaler.fit_transform(masked_map)
    n_rows, n_columns = 100, 100
    try:
        import _pickle as pkl
        init_codebook = pkl.load(open("somoclu_codebook.pkl", "rb" ))
        logging.info(f"Loaded the previous Codebook with {som.codebook.shape}")
    except:
        pass


    logging.info(f"SOM on unmasked regions")
    som = somoclu.Somoclu(n_columns, n_rows,
                           compactsupport=True,
                           initialization="pca",
                           neighborhood="gaussian",
                          #  initialcodebook=init_codebook,
                           std_coeff=0.5, 
                           verbose=2)
    som.train(clean_map, epochs=10, 
              radius0=0, radiusN=1, 
              radiuscooling='linear', scale0=0.1, 
              scaleN=0.01, scalecooling='linear')

    logging.info(f"Pickling BMUS and Codebooks")
    logging.info(som.bmus.shape)
    logging.info(som.codebook.shape)


    with open('somoclu100_bmus.pkl', 'wb') as file1:
        import _pickle as pkl
        pkl.dump(som.bmus, file1, -1)

    with open('somoclu100_codebook.pkl', 'wb') as file2:
        import _pickle as pkl
        pkl.dump(som.codebook, file2, -1)

    logging.info(f"SOM on masked regions")
    som1 = somoclu.Somoclu(n_columns, n_rows,
                           compactsupport=True,
                           initialization="pca",
                           neighborhood="gaussian",
                          #  initialcodebook=init_codebook,
                           std_coeff=0.5, 
                           verbose=2)

    som1.train(masked_map, epochs=10, 
              radius0=0, radiusN=1, 
              radiuscooling='linear', scale0=0.1, 
              scaleN=0.01, scalecooling='linear')

    logging.info(f"Pickling BMUS and Codebooks")
    logging.info(som1.bmus.shape)
    logging.info(som1.codebook.shape)


    with open('somoclu100_masked_bmus.pkl', 'wb') as file1:
        import _pickle as pkl
        pkl.dump(som1.bmus, file1, -1)

    with open('somoclu100_masked_codebook.pkl', 'wb') as file2:
        import _pickle as pkl
        pkl.dump(som1.codebook, file2, -1)


    for i in range(100):
        finalmap = np.full(len(maps),hp.UNSEEN)
        logging.info(f"Clustering with Random Seed State {i}")
        clustermethod = clus.MiniBatchKMeans(n_clusters=kk, n_init = 10, random_state=i)
        som.cluster(algorithm=clustermethod)
        data_cluster = []
        error = []
        for k in range(len(som.bmus)):
            a,b = som.bmus[k]
            data_cluster.append(som.clusters[b][a])
            temp = np.dot(som.codebook[b][a],maps[k])
            error.append(temp)
        error = np.array(error)
        data_cluster = np.array(data_cluster)
        finalmap[notmasked] = data_cluster

        som1.cluster(algorithm=clustermethod)
        data_cluster1 = []
        error1 = []
        for k in range(len(som.bmus)):
            a,b = som.bmus[k]
            data_cluster.append(som1.clusters[b][a])
            temp = np.dot(som1.codebook[b][a],maps[k])
            error.append(temp)
        error = np.array(error)
        data_cluster1 = np.array(data_cluster1) + 1
        finalmap[masked] = -data_cluster1


        logging.info(f"Cluster Shape {data_cluster.shape} and {data_cluster1.shape}")
        hp.write_map(f"somoclu-cluster_scaled_{i}.fits", finalmap, overwrite=True)


def load_somoclu(maps, COUNTER, kk=13):
    dim = 10 # int(np.sqrt(len(maps)))
    som = somoclu.Somoclu(dim, dim, data=maps, initialization="pca", verbose=2)
    som.load_bmus(f"som-bmus_{dim}")
    logging.info(som.bmus.shape)
    clustermethod = clus.MiniBatchKMeans(n_clusters=kk, n_init = 10, random_state=COUNTER)
    som.cluster(algorithm=clustermethod)
    return som.clusters


def handle_exception(exc_type, exc_value, exc_traceback):
    """ Handle Exceptions in log."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception",
                  exc_info=(exc_type, exc_value, exc_traceback))

class LoggerWriter(object):
    def __init__(self, writer):
        self._writer = writer
        self._msg = ''

    def write(self, message):
        self._msg = self._msg + message
        while '\n' in self._msg:
            pos = self._msg.find('\n')
            self._writer(self._msg[:pos])
            self._msg = self._msg[pos+1:]

    def flush(self):
        if self._msg != '':
            self._writer(self._msg)
            self._msg = ''


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--res", help="Resolution", type=int, 
                        default=60)
    parser.add_argument("--log", help="Set Log Level", type=str,
                        default='INFO')
    args = parser.parse_args()
    res = args.res
    loglevel = args.log.upper()
    logging.basicConfig(format='(%(asctime)s %(filename)s %(levelname)s) '
                        + '%(funcName)s %(lineno)d >> %(message)s',
                        filename=f"thelog100_{res}.log",
                        filemode='w',
                        level=getattr(logging, loglevel, None))
    logging.captureWarnings(True)
    log = logging.getLogger()
    sys.stdout = LoggerWriter(log.info)
    sys.stderr = LoggerWriter(log.warning)
    sys.excepthook = handle_exception
    maps = read_maps(res)
    logging.info(maps.shape)
    logging.info(maps)
    logging.info("Doing SOM")
    somoclu_cluster(maps)
    logging.info("Exiting Program")
