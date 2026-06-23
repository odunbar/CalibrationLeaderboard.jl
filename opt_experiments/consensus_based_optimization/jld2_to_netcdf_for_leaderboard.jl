using NCDatasets



function write_results_nc(
    filename;
    random_seed,
    ensemble_size,
    rmse_target,
    algorithm_type,
    metric,
)

    ds = Dataset(filename, "c")

    
    minimum_required_names = ["random_seed", "ensemble_size", "rmse_target", "algorithm_type"]
    #
    # Define dimensions from actual data
    #
    defDim(ds, "random_seed", length(random_seed))
    defDim(ds, "ensemble_size", length(ensemble_size))
    defDim(ds, "rmse_target", length(rmse_target))
    defDim(ds, "algorithm_type", length(algorithm_type))

    #
    # Coordinate variables
    #
    v_seed = defVar(ds, "random_seed", Int64, ("random_seed",))

    v_ens = defVar(ds, "ensemble_size", Float64, ("ensemble_size",))
    v_ens.attrib["description"] = "Number of ensemble members"

    v_rmse = defVar(ds, "rmse_target", Float64, ("rmse_target",))
    v_rmse.attrib["description"] =
        "Target accuracy level (root mean square error)"

    v_alg = defVar(ds, "algorithm_type", String, ("algorithm_type",))

    #
    # Main variable
    #
    v_metric = defVar(
        ds,
        "metric",
        Float64,
        ("random_seed", "ensemble_size", "rmse_target", "algorithm_type"),
        fillvalue = NaN,
    )

    v_metric.attrib["description"] =
        "Number of forward model evaluations (i.e., algorithm cost)"

    #
    # Write data
    #
    v_seed[:] = random_seed
    v_ens[:] = ensemble_size
    v_rmse[:] = rmse_target
    v_alg[:] = algorithm_type

    v_metric[:] = metric

    close(ds)
end
