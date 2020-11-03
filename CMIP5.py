import multiprocessing


def rmse(predictions, targets):
    import numpy as np
    return np.sqrt(((predictions - targets) ** 2).mean())


def check_dataset(xarray_dataset):
    '''A serie of checks to make sure that the dataset can be processed by the scripts'''

    model = xarray_dataset.attrs["source_id"]
    if 'latitude' in xarray_dataset.coords:
        try:
            xarray_dataset = xarray_dataset.rename({'longitude': 'lon', 'latitude': 'lat'})
        except:
            print("Couldn't find the right syntax for coordinates")
            return None
    if model == 'ERA5' :
        xarray_dataset = xarray_dataset.sel(expver=1)
    else : pass
    try:
        _x = xarray_dataset.sel(time=slice("1979-01-01", '2005-12-31'))
        if len(_x.time) < 300:
            print("Time slice not equal to 324 for{}".format(model))
            return None
        else:
            return _x
    except:
        print('time slice not matching for {}'.format(model))


def xa_process(paths, var , lat, lon):
    try:

        _x = xr.open_mfdataset(paths)
        if 'model_id' in _x.attrs:
            model = _x.attrs['model_id']
        else:
            era_dic = {'pr': 'tp', 'tas': 't2m', 'ps': 'sp'}
            var = era_dic[var]
            model = "ERA5"
        _x = _x[var]
        _x.attrs["source_id"] = model
        valid_dataset = check_dataset(_x)
        if valid_dataset is None:
            return None
        else:
            return regrid(valid_dataset).sel(lat=slice(*lat), lon=slice(*lon))
    except:
        print('issues xa_process')
        return None


def regrid(ds):
    """
    Define common grid and use xESMF to regrid all datasets
    returns:
        data_2x2: a list of datasets on the common grid
    """
    # regrid all lon,lat data to a common 2x2 grid
    import xesmf as xe
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 2.5)),
                         'lon': (['lon'], np.arange(-180, 180, 2.5)),
                         })

    if 'source_id' in ds.attrs:
        print('Regridding {}'.format(ds.attrs['source_id']))
    else:
        pass

    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=True, reuse_weights=True)
    _t = regridder(ds)
    _t.attrs = ds.attrs

    return _t


def xar_geoquery(xar, lat, lon):
    return xar.sel(lat=slice(*lat), lon=slice(*lon)).mean(dim=['lat', 'lon'])


def get_url(ds):
    try:
        files = ds.file_context().search()  # use "len(files)" to see total files found
        return [i.opendap_url for i in files if i.opendap_url is not None]
    except:
        print("something went wrong")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import xarray as xr
    import glob, os
    import warnings
    from itertools import product
    from sklearn import preprocessing
    from pyesgf.search import SearchConnection
    from itertools import product

    warnings.filterwarnings("ignore")

    # os.chdir('/home/dump')
    lon = [-130, -100]
    lat = [60, 70]
    era_nc = glob.glob("ada*.nc")

    time = pd.date_range(start='01/01/1979', end='12/31/2005', freq='MS')

    variables = [["pr", "tp"] , ["tas", "t2m"], ["ps", "sp"]]

    distrib = True
    CEDA_SERVICE = 'https://esgf-node.llnl.gov//esg-search'
    conn = SearchConnection(CEDA_SERVICE, distrib=distrib)
    ctx = conn.new_context(latest=True,
                           project='CMIP5',
                           experiment='historical',
                           time_frequency="mon",
                           realm='atmos',
                           ensemble='r1i1p1')

    # ctx2 = ctx.constrain(
    #     latest=True,
    #     project='CMIP5',
    #     experiment='historical',
    #     time_frequency="mon",
    #     realm='atmos',
    #     ensemble='r1i1p1')

    ds = ctx.search()
    print("ctx2 is {}".format(ctx.hit_count))
    with multiprocessing.Pool(processes=16) as pool:
        url_list = pool.map(get_url, ds)

    url_list = [i for i in url_list if i is not None]
    flat = [y for x in url_list for y in x]
    flat = [f for f in flat if f is not None]

    ######################## For test only
    # v = 'tas'
    d = {}
    for var in variables[1:2]:
        v = var[0]
        d[v] = {}
        paths = [u for u in flat if "/{}/".format(v) in u]

        models = set([n.split("_")[-4] for n in paths])

        if distrib == True:
            for m in models:
                for i in [p for p in paths if m in p]:
                    server = i.split('/')[2]
                    d[v][m] = {server: [i]}

            # d = {model: [f for f in paths if model in f] for model in models}
            t = [[d[v][m][s] for s in d[v][m].keys()] for m in d[v].keys()]
            t += [era_nc]
            d[v]['ERA5'] = {'ERA5' : era_nc}

            # with multiprocessing.Pool(processes=8) as pool:
            #     ds_l = pool.starmap(xa_process, product(t, [v]))

            for m in d[v].keys():
                if m != 'CSIRO-Mk3-6-0' :
                    for s in d[v][m].keys():
                        print('working on {}{}{}'.format(v, m, s))
                        d[v][m][s] = xa_process(d[v][m][s], v , lat , lon)
                        # if d[v][m][s] is not None :
                        #     d[v][m][s] = xar_geoquery(d[v][m][s], lat, lon).values
            # ds_l = [_d for _d in ds_l if _d is not None]


        else:
            d[v] = {model: [f for f in paths if model in f] for model in models}

            # with multiprocessing.Pool(processes=8) as pool:
            #     ds_l = pool.starmap(xa_process, product([d[v][m] for m in d[v].keys()], [v]))

            ds_l = []
            for m in d[v].keys():
                ds_l += [xa_process(d[v][m], v)]

            ds_l = [_d for _d in ds_l if _d is not None]  # there must be a better way to return nothing from a function
            _t = {model: xar_geoquery(XarArray, lat, lon).values for model, XarArray in ds_l}
            df = pd.DataFrame(_t, index=time)
            # df = pd.DataFrame({model: xar_geoquery(XarArray, lat, lon) for model, XarArray in ds_l}, index=time)

            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(df)
            _df = pd.DataFrame(np_scaled, columns=df.columns, index=time)

            for m in df.columns:
                if m != "ERA5":
                    a = []
                    try:
                        for i in range(1, 13):
                            r = rmse(_df[m][_df.index.month == i].values, _df['ERA5'][_df.index.month == i].values)
                            a += [r]
                    except:
                        print("{} failed".format(m))
                    d[v][m] = sum(a)



    res = {}
    for v in d.keys() :
        res[v] = {}
        for m in d[v].keys():
            if len(d[v][m].keys()) > 1 :
                for s in d[v][m][s].keys() :
                    if d[v][m][s] is None :
                        pass
                    elif d[v][m][s].mean() == 0 :
                        pass
                    else :
                        res[v][m] = d[v][m][s]
            else :
                s = list(d[v][m].keys())[0]
                if d[v][m][s] is not None :
                    res[v][m] = d[v][m][s]









    da = pd.DataFrame(d)
    da.dropna(inplace=True)
    da['sum'] = da.sum(axis=1)
    da['rank'] = da['sum'].rank()
    da.sort_values('rank', inplace=True)

    da.to_csv('Results_CMIp5_analysis.csv')

    # for var in variables:
    #     v = var[0]
    #     d={}
    #     d[v] = {}
    #     paths = [u for u in flat if "/{}/".format(v) in u]
    #
    #     d = {model: [f for f in paths if model in f] for model in models}
    #
    #     for m in models :
    #         for i in [p for p in paths if m in p]:
    #             server = i.split('/')[2]
    #             d[v][m] = {server : i}
    #
    #     # d = {model: [f for f in paths if model in f] for model in models}
    #     t= [[d[v][m][s] for s in d[v][m].keys()]for m in d[v].keys()]
    #     t += [era_nc]
    #     # d['ERA5'] = era_nc
    #
    #     with multiprocessing.Pool(processes=8) as pool:
    #         ds_l = pool.starmap(xa_process, product(t, [v]))
    #
    #     with multiprocessing.Pool(processes=8) as pool:
    #         ds_l = pool.starmap(xa_process, product([d[m] for m in d.keys()], [v]))
    #
    #     ds_l = [_d for _d in ds_l if _d is not None]  # there must be a better way to return nothing from a function
    #
    #     df = pd.DataFrame({model: xar_geoquery(XarArray, lat, lon) for model, XarArray in ds_l}, index=time)
    #
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     np_scaled = min_max_scaler.fit_transform(df)
    #     _df = pd.DataFrame(np_scaled, columns=df.columns, index=time)
    #
    #     for m in df.columns:
    #         if m != "ERA5":
    #             a = []
    #             for i in range(1, 13):
    #                 r = rmse(_df[m][_df.index.month == i].values, _df['ERA5'][_df.index.month == i].values)
    #                 a += [r]
    #             d[v][m] = sum(a)
    #
    # da = pd.DataFrame(d)
    # da.dropna(inplace=True)
    # da['sum'] = da.sum(axis=1)
    # da['rank'] = da['sum'].rank()
    # da.sort_values('rank', inplace=True)
    #
    # da.to_csv('Results_CMIP6_analysis.csv')
    # # with open('CMIP6_NWT_ERA6.pickle', 'wb') as handle:
    # #     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # #
    # # d = pd.read_pickle('CMIP6_NWT_ERA6.pickle')
