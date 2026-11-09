# # Node/DOF maps and coloring support.

function _cell_nodes(el::ReferenceFiniteElement, active_bg_cells, nx::Int)
    nodes = Matrix{Int32}(undef, nnode(el), length(active_bg_cells))
    for (cell, bg_cell) in enumerate(active_bg_cells)
        nodes[:, cell] .= cell_nodes(el, Int(bg_cell), nx)
    end
    return nodes
end

function _effective_cell_colors(test_dofs, trial_dofs)
    colors = Vector{Vector{Int32}}()
    used_dofs = BitSet[]
    for cell in eachindex(test_dofs)
        selected = 0
        for color in eachindex(colors)
            conflict = false
            for dof in test_dofs[cell]
                if dof in used_dofs[color]
                    conflict = true
                    break
                end
            end
            if !conflict
                for dof in trial_dofs[cell]
                    if dof in used_dofs[color]
                        conflict = true
                        break
                    end
                end
            end
            if !conflict
                selected = color
                break
            end
        end
        if selected == 0
            push!(colors, Int32[])
            push!(used_dofs, BitSet())
            selected = length(colors)
        end
        push!(colors[selected], Int32(cell))
        union!(used_dofs[selected], test_dofs[cell])
        union!(used_dofs[selected], trial_dofs[cell])
    end
    return colors
end
