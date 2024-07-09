using MAT

function read_and_print_mat_files(folder_path, file_prefix, num_files)
    #for i in 1:num_files
        i = num_files
        file_num = lpad(i, 2, '0')  # Asegura que el número tenga dos dígitos
        file_path = joinpath(folder_path, "$file_prefix$file_num.mat")
        
        # Check if the file exists
        if isfile(file_path)
            println("Reading file: $file_path")
            matfile = matread(file_path)
            
            # Print the content of the .mat file
            for (name, value) in matfile
                println("Variable: $name")
                if(name == "R50")
                    println("Value: $value")
                end
                println()
            end
        else
            println("File not found: $file_path")
        end
    #end
end

# Specify the folder path and file prefix
folder_path = "test/matlab"
file_prefix = "f"
num_files = 13

# Call the function to read and print .mat files
read_and_print_mat_files(folder_path, file_prefix, num_files)
