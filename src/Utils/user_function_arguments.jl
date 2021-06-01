@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{1}) =
    @inbounds (ℑ[1](i, j, k, grid, model_fields[idx[1]]),)

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{2}) =
    @inbounds (ℑ[1](i, j, k, grid, model_fields[idx[1]]),
               ℑ[2](i, j, k, grid, model_fields[idx[2]]))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{3}) =
    @inbounds (ℑ[1](i, j, k, grid, model_fields[idx[1]]),
               ℑ[2](i, j, k, grid, model_fields[idx[2]]),
               ℑ[3](i, j, k, grid, model_fields[idx[3]]))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, idx::NTuple{N}) where N =
    @inbounds ntuple(n -> ℑ[n](i, j, k, grid, model_fields[idx[n]]), Val(N))

""" Returns field arguments in user-defined functions for forcing and boundary conditions."""
@inline function user_function_arguments(i, j, k, grid, user_func, model_fields, ::Nothing, args...)

    ℑ = user_func.field_dependencies_interp
    idx = user_func.field_dependencies_indices
    field_args = field_arguments(i, j, k, grid, model_fields, ℑ, idx)

    return return tuple(field_args..., args...)
end

""" Returns field arguments plus parameters in user-defined functions for forcing and boundary conditions."""
@inline function user_function_arguments(i, j, k, grid, user_func, model_fields, parameters, args...)

    ℑ = user_func.field_dependencies_interp
    idx = user_func.field_dependencies_indices
    parameters = user_func.parameters

    field_args = field_arguments(i, j, k, grid, model_fields, ℑ, idx)

    return tuple(field_args..., parameters, args...)
end

