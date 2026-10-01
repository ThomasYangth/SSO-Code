#=
Persistence: HDF5 MPS files named from their metadata, plus per-model
"tracking" CSVs (one row per converged result, theta excluded from the
filename) that make parameter scans resumable. `run_dmrg_for_size` consults
`load_existing_file` before every solve and `saveMPS` after it (unless
`save_mps=false`, which the ε-ladder drivers use: they write their own
`done_stepNN.h5` / `rolling_stepNN.h5` checkpoints instead).
=#

# MPSMetadata is now just a Dict from alphabetic strings to numbers
const MPSMetadata = Dict{String, Number}

function validate_metadata_key(key::String)::Bool
    return occursin(r"^[a-zA-Z]+$", key)
end

function validate_metadata(metadata::MPSMetadata)::Bool
    for (key, value) in metadata
        if !validate_metadata_key(key)
            @warn "Invalid metadata key '$key': must contain only letters"
            return false
        end
        if !(value isa Number)
            @warn "Invalid metadata value for key '$key': must be a number"
            return false
        end
    end
    return true
end

function encode_filename(metadata::MPSMetadata)::String
    if !validate_metadata(metadata)
        error("Invalid metadata for filename encoding")
    end
    
    # Sort keys for consistent filename generation
    sorted_keys = sort(collect(keys(metadata)))
    
    filename_parts = ["MPS"]
    for key in sorted_keys
        value = metadata[key]
        # Format numbers to avoid unnecessary decimals
        value_str = isa(value, Integer) ? string(value) : string(Float64(value))
        push!(filename_parts, "$(key)$(value_str)")
    end
    
    return join(filename_parts, "_") * ".h5"
end

"""
Create tracking filename based on metadata (excludes theta if present)
"""
function encode_tracking_filename(metadata::MPSMetadata)::String
    if !validate_metadata(metadata)
        error("Invalid metadata for filename encoding")
    end
    
    # Exclude theta from tracking filename
    tracking_metadata = copy(metadata)
    delete!(tracking_metadata, "theta")  # Remove theta if present
    
    # Sort keys for consistent filename generation
    sorted_keys = sort(collect(keys(tracking_metadata)))
    
    filename_parts = ["tracking"]
    for key in sorted_keys
        value = tracking_metadata[key]
        # Format numbers to avoid unnecessary decimals
        value_str = isa(value, Integer) ? string(value) : string(Float64(value))
        push!(filename_parts, "$(key)$(value_str)")
    end
    
    return join(filename_parts, "_") * ".csv"
end

"""
Convert various data types to string representation for CSV storage
"""
function convert_data_for_csv(value)::String
    if value isa Number
        return string(value)
    elseif value isa AbstractVector
        # Convert vector to comma-separated string enclosed in brackets
        return "[" * join(string.(value), ",") * "]"
    elseif value isa AbstractMatrix
        # Convert matrix to string representation
        return "[" * join([join(string.(row), ",") for row in eachrow(value)], ";") * "]"
    else
        return string(value)
    end
end

"""
Parse string representation back to original data type
"""
function parse_data_from_csv(value_str::AbstractString)
    if isempty(value_str)
        return missing
    end

    # Convert to String to handle String31 and other subtypes
    value_str = String(value_str)

    # Check if it's a vector/matrix (starts and ends with brackets)
    if startswith(value_str, "[") && endswith(value_str, "]")
        inner = value_str[2:end-1]  # Remove brackets

        if contains(inner, ";")
            # Matrix format: [1,2;3,4]
            rows = split(inner, ";")
            matrix_data = []
            for row_str in rows
                if !isempty(row_str)
                    row_vals = [parse_single_number(strip(x)) for x in split(row_str, ",") if !isempty(strip(x))]
                    push!(matrix_data, row_vals)
                end
            end
            if !isempty(matrix_data)
                # Convert to matrix, assuming all rows have same length
                return hcat(matrix_data...)'
            end
        else
            # Vector format: [1,2,3]
            if !isempty(inner)
                return [parse_single_number(strip(x)) for x in split(inner, ",") if !isempty(strip(x))]
            else
                return Float64[]
            end
        end
    else
        # Single number or string
        return parse_single_number(value_str)
    end

    return value_str  # Fallback to string
end

"""
Try to parse a string as Int, then Float64, otherwise return as string
"""
function parse_single_number(s::AbstractString)
    s = strip(s)
    if isempty(s)
        return missing
    end
    
    try
        # Check for scientific notation or decimal points
        if contains(lowercase(s), "e") || contains(s, ".")
            return parse(Float64, s)
        else
            # Try integer for whole numbers
            return parse(Int, s)
        end
    catch
        return s  # Return as string if parsing fails
    end
end

"""
Update or create tracking file with new entry
"""
function update_tracking_file(metadata::MPSMetadata, other_data::Dict, directory::String)
    tracking_filename = encode_tracking_filename(metadata)
    tracking_path = joinpath(directory, tracking_filename)
    
    # Prepare the new row data
    timestamp = string(Dates.now())
    new_row = Dict{String, String}()
    new_row["timestamp"] = timestamp
    
    # Add theta if present in metadata
    if haskey(metadata, "theta")
        new_row["theta"] = convert_data_for_csv(metadata["theta"])
    end

    # Add all other_data entries
    for (key, value) in other_data
        new_row[key] = convert_data_for_csv(value)
    end

    # Check if tracking file exists
    if isfile(tracking_path)
        # Load existing data with all string types
        existing_df = CSV.read(tracking_path, DataFrame; types=String)
        
        # Get all column names (union of existing and new)
        all_columns = union(names(existing_df), keys(new_row))
        
        # Add missing columns to existing dataframe
        for col in all_columns
            if !(col in names(existing_df))
                existing_df[!, col] = fill("", nrow(existing_df))
            end
        end
        
        # Create new row with all columns as vector
        new_row_complete = []
        for col in all_columns
            push!(new_row_complete, get(new_row, col, ""))
        end
        
        # Add new row using vector
        push!(existing_df, new_row_complete)
        
        # Write updated dataframe
        CSV.write(tracking_path, existing_df)
    else
        # Create new tracking file
        new_df = DataFrame(new_row)
        CSV.write(tracking_path, new_df)
    end
    
    println("Updated tracking file: $tracking_path")
end

"""
Load tracking file based on metadata (excludes theta) and return as DataFrame
If parse_types=true, converts string representations back to original data types
"""
function load_tracking_file(metadata::MPSMetadata; directory::String = data_dir(), parse_types::Bool = true)::DataFrame
    tracking_filename = encode_tracking_filename(metadata)
    tracking_path = joinpath(directory, tracking_filename)
    
    if !isfile(tracking_path)
        @warn "Tracking file not found: $tracking_path"
        return DataFrame()
    end
    
    try
        df = CSV.read(tracking_path, DataFrame)
        
        if parse_types
            # Parse columns back to original data types (except timestamp)
            for col_name in names(df)
                if col_name != "timestamp"
                    df[!, col_name] = [parse_data_from_csv(string(x)) for x in df[!, col_name]]
                end
            end
        end
        
        println("Loaded tracking file: $tracking_path")
        println("Found $(nrow(df)) entries")
        return df
    catch e
        @warn "Failed to load tracking file $tracking_path: $e"
        return DataFrame()
    end
end

"""
Load all tracking files in directory and return as dictionary mapping filename to DataFrame
If parse_types=true, converts string representations back to original data types
"""
function load_all_tracking_files(; directory::String = data_dir(), parse_types::Bool = true)::Dict{String, DataFrame}
    if !isdir(directory)
        @warn "Directory $directory does not exist"
        return Dict{String, DataFrame}()
    end
    
    tracking_files = Dict{String, DataFrame}()
    
    for file in readdir(directory)
        if startswith(file, "tracking_") && endswith(file, ".csv")
            full_path = joinpath(directory, file)
            try
                df = CSV.read(full_path, DataFrame)
                
                if parse_types
                    # Parse columns back to original data types (except timestamp)
                    for col_name in names(df)
                        if col_name != "timestamp"
                            df[!, col_name] = [parse_data_from_csv(string(x)) for x in df[!, col_name]]
                        end
                    end
                end
                
                tracking_files[file] = df
                println("Loaded tracking file: $file ($(nrow(df)) entries)")
            catch e
                @warn "Failed to load tracking file $file: $e"
            end
        end
    end
    
    println("Loaded $(length(tracking_files)) tracking files")
    return tracking_files
end

"""
Check if a new entry is already present in the existing DataFrame
Compares all columns except timestamp
"""
function is_duplicate_entry(new_entry::Dict{String, String}, existing_df::DataFrame)::Bool
    if nrow(existing_df) == 0
        return false
    end
    
    # Get columns to compare (all except timestamp)
    compare_cols = [col for col in keys(new_entry) if col != "timestamp"]
    
    for row_idx in 1:nrow(existing_df)
        is_match = true
        for col in compare_cols
            if col in names(existing_df)
                existing_val = string(existing_df[row_idx, col])
                new_val = new_entry[col]
                if existing_val != new_val
                    is_match = false
                    break
                end
            else
                # Column doesn't exist in existing data, check if new value is empty
                if !isempty(new_entry[col])
                    is_match = false
                    break
                end
            end
        end
        
        if is_match
            return true
        end
    end
    
    return false
end

"""
Migrate existing .h5 MPS files to tracking CSV files
Only adds entries that don't already exist in the tracking files

Parameters:
- directory: Directory to scan for .h5 files
- dry_run: If true, only reports what would be done without actually writing files
"""
function migrate_h5_to_tracking(; directory::String = data_dir(), dry_run::Bool = false)
    if !isdir(directory)
        @warn "Directory $directory does not exist"
        return
    end
    
    h5_files = [f for f in readdir(directory) if endswith(f, ".h5") && startswith(f, "MPS_")]
    
    if isempty(h5_files)
        println("No MPS .h5 files found in $directory")
        return
    end
    
    println("Found $(length(h5_files)) MPS files to process")
    
    migration_stats = Dict("processed" => 0, "added" => 0, "duplicates" => 0, "errors" => 0)
    
    for filename in h5_files
        migration_stats["processed"] += 1
        full_path = joinpath(directory, filename)
        
        try
            # Load metadata and other_data from .h5 file
            f = h5open(full_path, "r")
            
            # Load metadata
            metadata = MPSMetadata()
            if haskey(f, "metadata")
                metadata_group = f["metadata"]
                for key in keys(metadata_group)
                    try
                        raw_value = read(metadata_group, key)
                        # Handle type conversion issues - ensure it's a Number
                        if raw_value isa AbstractString
                            # Try to parse string as number
                            parsed_value = parse_single_number(raw_value)
                            if parsed_value isa Number
                                metadata[key] = parsed_value
                            else
                                @warn "Metadata key '$key' has non-numeric string value: '$raw_value'"
                                continue
                            end
                        elseif raw_value isa Number
                            metadata[key] = raw_value
                        else
                            @warn "Metadata key '$key' has unsupported type: $(typeof(raw_value))"
                            continue
                        end
                    catch e
                        @warn "Error reading metadata key '$key' from $filename: $e"
                        continue
                    end
                end
            else
                # Fall back to filename parsing
                try
                    metadata = decode_filename(filename)
                catch e
                    @warn "Could not decode metadata from filename $filename: $e"
                    close(f)
                    migration_stats["errors"] += 1
                    continue
                end
            end

            # Default flq to 0 if not present (for backward compatibility)
            if !haskey(metadata, "flq")
                metadata["flq"] = 0
            end

            # Load timestamp
            file_timestamp = haskey(f, "timestamp") ? read(f, "timestamp") : ""
            if isempty(file_timestamp)
                # Use file modification time as fallback
                file_timestamp = string(Dates.unix2datetime(stat(full_path).mtime))
            end
            
            # Load other_data (everything that's not psi, metadata, timestamp, or comments)
            other_data = Dict()
            for key in keys(f)
                if !(key in ["psi", "metadata", "timestamp", "comments"])
                    other_data[key] = read(f, key)
                end
            end
            
            close(f)
            
            # Skip files with conv != 1
            conv_value = get(metadata, "conv", nothing)
            if conv_value != 1
                println("  Skipping $filename - conv != 1 (conv = $conv_value)")
                continue
            end
            
            # Prepare entry for tracking file
            new_row = Dict{String, String}()
            new_row["timestamp"] = file_timestamp
            
            # Add theta if present in metadata
            if haskey(metadata, "theta")
                new_row["theta"] = convert_data_for_csv(metadata["theta"])
            end
            
            # Add all other_data entries
            for (key, value) in other_data
                new_row[key] = convert_data_for_csv(value)
            end
            
            # Get tracking file path
            tracking_filename = encode_tracking_filename(metadata)
            tracking_path = joinpath(directory, tracking_filename)
            
            # Check if entry already exists
            entry_exists = false
            if isfile(tracking_path)
                existing_df = CSV.read(tracking_path, DataFrame)
                entry_exists = is_duplicate_entry(new_row, existing_df)
            end
            
            if entry_exists
                println("  Skipping $filename - entry already exists in $tracking_filename")
                migration_stats["duplicates"] += 1
            else
                if dry_run
                    println("  Would add $filename to $tracking_filename")
                    migration_stats["added"] += 1
                else
                    # Add entry to tracking file
                    if isfile(tracking_path)
                        # Load existing and append
                        existing_df = CSV.read(tracking_path, DataFrame; types=String)
                        
                        # Get all column names (union of existing and new)
                        all_columns = union(names(existing_df), keys(new_row))
                        
                        # Add missing columns to existing dataframe
                        for col in all_columns
                            if !(col in names(existing_df))
                                existing_df[!, col] = fill("", nrow(existing_df))
                            end
                        end
                        
                        # Create new row with all columns
                        new_row_complete = []
                        for col in all_columns
                            push!(new_row_complete, get(new_row, col, ""))
                        end
                        
                        # Add new row using push! with vector
                        push!(existing_df, new_row_complete)
                        CSV.write(tracking_path, existing_df)
                    else
                        # Create new tracking file
                        new_df = DataFrame(new_row)
                        CSV.write(tracking_path, new_df)
                    end
                    
                    println("  Added $filename to $tracking_filename")
                    migration_stats["added"] += 1
                end
            end
            
        catch e
            @warn "Error processing $filename: $e"
            migration_stats["errors"] += 1
            continue
        end
    end
    
    # Print summary
    println("\n" * "="^50)
    println("MIGRATION SUMMARY")
    println("="^50)
    println("Files processed: $(migration_stats["processed"])")
    println("Entries added: $(migration_stats["added"])")
    println("Duplicates skipped: $(migration_stats["duplicates"])")
    println("Errors encountered: $(migration_stats["errors"])")
    
    if dry_run
        println("\nThis was a DRY RUN - no files were actually modified")
        println("Run with dry_run=false to perform the actual migration")
    end
    
    return migration_stats
end

function decode_filename(filename::String)::MPSMetadata
    basename_file = basename(filename)
    if !startswith(basename_file, "MPS_") || !endswith(basename_file, ".h5")
        error("Invalid MPS filename format: $filename")
    end
    
    # Remove "MPS_" prefix and ".h5" suffix
    basename_file = basename_file[5:end-3]
    parts = split(basename_file, "_")
    
    metadata = MPSMetadata()
    
    for part in parts
        # Use regex to separate alphabetic key from numeric value
        match_result = match(r"^([a-zA-Z]+)(.+)$", part)
        if match_result !== nothing
            key = match_result.captures[1]
            value_str = match_result.captures[2]

            # Try to parse as Int first, then Float64
            try
                if occursin(".", value_str) || occursin("e", lowercase(value_str))
                    value = parse(Float64, value_str)
                else
                    value = parse(Int, value_str)
                end
                metadata[key] = value
            catch
                @warn "Could not parse value '$value_str' for key '$key' in filename: $filename"
            end
        else
            @warn "Could not parse filename part '$part' in: $filename"
        end
    end

    # Default flq to 0 if not present (for backward compatibility)
    if !haskey(metadata, "flq")
        metadata["flq"] = 0
    end

    return metadata
end

function saveMPS(psi::MPS, metadata::MPSMetadata; 
                directory::String = data_dir(), 
                comments::String = "",
                other_data::Dict=Dict())
    
    if !validate_metadata(metadata)
        error("Invalid metadata")
    end
    
    # Generate filename from metadata
    filename = encode_filename(metadata)
    full_path = joinpath(directory, filename)
    
    # Ensure directory exists
    if !isdir(directory)
        mkpath(directory)
    end
    
    try
        f = h5open(full_path, "w")
        
        # Save the MPS state (assuming psi has a cpuarray method or similar)
        write(f, "psi", psi)
        
        # Save all numeric metadata
        metadata_group = create_group(f, "metadata")
        for (key, value) in metadata
            write(metadata_group, key, value)
        end
        
        # Save non-numeric data
        write(f, "timestamp", string(Dates.now()))
        if !isempty(comments)
            write(f, "comments", comments)
        end

        for (key, value) in other_data
            write(f, key, value)
        end
        
        close(f)
        println("MPS saved successfully to: $full_path")
        
        # Update tracking file only if conv == 1
        conv_value = get(metadata, "conv", nothing)
        if conv_value == 1
            try
                update_tracking_file(metadata, other_data, directory)
            catch e
                @warn "Failed to update tracking file: $e"
            end
        else
            println("Skipping tracking file update - conv != 1 (conv = $conv_value)")
        end
        
        return full_path
        
    catch e
        error("Failed to save MPS: $e")
    end
end

function loadMPS(filename::String; directory::String = data_dir())

    filename = joinpath(directory, filename)

    if !isfile(filename)
        error("File $filename does not exist.")
    end
    
    try
        f = h5open(filename, "r")
        
        # Load the MPS state
        psi = read(f, "psi", MPS)
        
        # Load metadata
        metadata = MPSMetadata()
        if haskey(f, "metadata")
            metadata_group = f["metadata"]
            for key in keys(metadata_group)
                metadata[key] = read(metadata_group, key)
            end
        else
            # Fall back to parsing filename
            try
                metadata = decode_filename(filename)
            catch e
                @warn "Could not decode filename metadata: $e"
            end
        end

        # Default flq to 0 if not present (for backward compatibility)
        if !haskey(metadata, "flq")
            metadata["flq"] = 0
        end
        
        # Load additional info
        timestamp = haskey(f, "timestamp") ? read(f, "timestamp") : ""
        comments = haskey(f, "comments") ? read(f, "comments") : ""
        
        # Load other_data (everything that's not psi, metadata, timestamp, or comments)
        other_data = Dict()
        for key in keys(f)
            if !(key in ["psi", "metadata", "timestamp", "comments"])
                other_data[key] = read(f, key)
            end
        end

        close(f)

        println("MPS loaded successfully from: $filename")
        println("Timestamp: $timestamp")
        if !isempty(comments)
            println("Comments: $comments")
        end
        show_metadata(metadata)
        
        return psi, metadata, other_data, timestamp, comments
        
    catch e
        error("Failed to load MPS from $filename: $e")
    end
end

"""
Scan all MPS files in directory and return DataFrame with all metadata
"""
function load_all_metadata(directory::String = data_dir())::DataFrame
    
    if !isdir(directory)
        @warn "Directory $directory does not exist"
        return DataFrame()
    end
    
    all_metadata = []
    filenames = []
    timestamps = []
    comments_list = []
    
    for file in readdir(directory)
        if endswith(file, ".h5") && startswith(file, "MPS_")
            full_path = joinpath(directory, file)
            
            try
                # Try to load from file first
                metadata = MPSMetadata()
                timestamp = ""
                comments = ""
                
                f = h5open(full_path, "r")
                
                if haskey(f, "metadata")
                    metadata_group = f["metadata"]
                    for key in keys(metadata_group)
                        metadata[key] = read(metadata_group, key)
                    end
                else
                    # Fall back to filename parsing
                    metadata = decode_filename(file)
                end

                # Default flq to 0 if not present (for backward compatibility)
                if !haskey(metadata, "flq")
                    metadata["flq"] = 0
                end

                timestamp = haskey(f, "timestamp") ? read(f, "timestamp") : ""
                comments = haskey(f, "comments") ? read(f, "comments") : ""

                close(f)

                push!(all_metadata, metadata)
                push!(filenames, file)
                push!(timestamps, timestamp)
                push!(comments_list, comments)
                
            catch e
                @warn "Could not load metadata from $file: $e"
                continue
            end
        end
    end
    
    if isempty(all_metadata)
        return DataFrame()
    end
    
    # Get all unique keys across all metadata dictionaries
    all_keys = Set{String}()
    for metadata in all_metadata
        union!(all_keys, keys(metadata))
    end
    all_keys = sort(collect(all_keys))
    
    # Create DataFrame
    df_data = Dict{String, Vector}()
    df_data["filename"] = filenames
    df_data["timestamp"] = timestamps
    df_data["comments"] = comments_list
    
    # Add columns for each metadata key
    for key in all_keys
        values = []
        for metadata in all_metadata
            push!(values, get(metadata, key, missing))
        end
        df_data[key] = values
    end
    
    return DataFrame(df_data)
end

# Utility functions
function show_metadata(metadata::MPSMetadata)
    println("=== MPS Metadata ===")
    sorted_keys = sort(collect(keys(metadata)))
    for key in sorted_keys
        println("$key: $(metadata[key])")
    end
    println("==================")
end

"""
Find MPS files matching given metadata criteria
"""
function find_files_by_metadata(criteria::MPSMetadata; directory::String = data_dir(), tolerance::Float64 = 1e-10)
    df = load_all_metadata(directory)
    
    if nrow(df) == 0
        return nothing
    end
    
    mask = trues(nrow(df))
    
    for (key, target_value) in criteria
        if key in names(df)
            column_values = df[!, key]
            if target_value isa AbstractFloat
                # Use tolerance for floating point comparisons
                mask .&= [!ismissing(v) && abs(v - target_value) < tolerance for v in column_values]
            else
                # Exact match for integers and other types
                mask .&= [!ismissing(v) && v == target_value for v in column_values]
            end
        else
            # If key doesn't exist, no files match
            return nothing
        end
    end
    
    return df[mask, :]
end


function load_existing_file(criteria::MPSMetadata; directory::String = data_dir(), tolerance::Float64 = 1e-10,
                            load_arb_init::Bool = false)
    """
    Load existing MPS file matching criteria.

    If load_arb_init=true, ignores the "init" field in criteria and searches for files
    with any init value. When multiple files exist with different init values, prioritizes
    the one with init closest to the target init (if specified in criteria).
    """

    # Separate criteria into init and non-init parts
    target_init = get(criteria, "init", nothing)
    search_criteria = if load_arb_init
        # Remove "init" from search criteria
        filter(p -> p.first != "init", criteria)
    else
        criteria
    end

    files = find_files_by_metadata(search_criteria; directory=directory, tolerance=tolerance)

    if files == nothing || nrow(files) == 0
        return -1, nothing
    end

    # Find the best file:
    # 1. Prioritize by conv (larger is better)
    # 2. If load_arb_init=true and multiple files have same conv, prioritize by init closest to target
    max_conv = -1
    best_filename = nothing
    best_init_distance = Inf

    for i in 1:nrow(files)
        if !ismissing(files[i, "conv"])
            current_conv = files[i, "conv"]
            current_init = get(files[i, :], "init", missing)

            # Calculate init distance if relevant
            init_distance = if load_arb_init && !isnothing(target_init) && !ismissing(current_init)
                abs(current_init - target_init)
            else
                0.0
            end

            # Update best file based on conv first, then init distance
            if current_conv > max_conv || (current_conv == max_conv && init_distance < best_init_distance)
                max_conv = current_conv
                best_filename = files[i, "filename"]
                best_init_distance = init_distance
            end
        end
    end

    if isnothing(best_filename)
        return -1, nothing
    else
        # Check if loaded file has different init than target (when load_arb_init=true)
        if load_arb_init && !isnothing(target_init)
            loaded_init = nothing
            for i in 1:nrow(files)
                if files[i, "filename"] == best_filename && !ismissing(files[i, "init"])
                    loaded_init = files[i, "init"]
                    break
                end
            end
            if !isnothing(loaded_init) && loaded_init != target_init
                @warn "Loading file with init=$loaded_init (target was init=$target_init)"
            end
        end
        return max_conv, loadMPS(best_filename; directory=directory)
    end

end