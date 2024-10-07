# first line: 403
    @memory.cache
    def _get_gldas_land_mask_cached(
        extent: Union[Tuple[float, float, float, float], str] = "global",
        verbose: bool = False,
    ) -> xr.DataArray:
        if verbose:
            print(
                f"Downloading GLDAS land mask from NASA...",
                end=" ",
                flush=True,
            )
        tic = time.time()

        fname = "GLDASp5_landmask_025d.nc4"
        url = "https://ldas.gsfc.nasa.gov/sites/default/files/ldas/gldas/VEG/" + fname
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            with open(fname, "wb") as f:
                f.write(response.read())
        da = xr.open_dataset(fname)["GLDAS_mask"].isel(time=0).drop("time").load()

        if isinstance(extent, str):
            extent = extent_str_to_tuple(extent)
        else:
            extent = tuple([float(x) for x in extent])
        lon_min, lon_max, lat_min, lat_max = extent

        # Reverse latitude to match ERA5
        da = da.reindex(lat=da.lat[::-1])
        da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        da.attrs = {}

        os.remove(fname)

        if verbose:
            print(f"{da.nbytes / 1e6:.2f} MB loaded in {time.time() - tic:.2f} s")

        return da
