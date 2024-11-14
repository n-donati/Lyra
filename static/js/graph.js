const width = window.innerWidth;
const height = window.innerHeight - 60;

const container = d3.select('#graph-container');

// Simplified container structure - remove wrapper
const canvas = container
    .append('canvas')
    .attr('width', width)
    .attr('height', height);

const context = canvas.node().getContext('2d');

const svg = container
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('position', 'absolute')
    .style('top', 0)
    .style('left', 0)
    .style('pointer-events', 'all');

const g = svg.append('g');

// Simple zoom
const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on('zoom', (e) => {
        g.attr('transform', e.transform);
        // Store transform for canvas
        currentTransform = e.transform;
        updateLabelsVisibility(currentTransform.k);
        requestAnimationFrame(render);
    });


function updateLabelsVisibility(zoomLevel) {
    // Escala personalizada para mostrar etiquetas basadas en el nivel de zoom
    d3.selectAll('.node-label')
        .style('display', d => {
            // Mostrar solo las etiquetas de los nodos grandes en zoom bajo, y más etiquetas en zoom alto
            if (zoomLevel > 1.2) {
                // Mostrar todas las etiquetas
                return 'block';
            } else if (zoomLevel > 0.7) {
                // Mostrar etiquetas solo de los nodos grandes
                return d.size > 10 ? 'block' : 'none';
            }else if (zoomLevel > 0.3) {
                // Mostrar etiquetas solo de los nodos grandes
                return d.size > 15 ? 'block' : 'none';
            } else {
                // Ocultar la mayoría de las etiquetas en zoom bajo
                return d.size > 20 ? 'block' : 'none';
            }
        });
}
    

svg.call(zoom);

let currentTransform = d3.zoomIdentity;
let nodes = [], links = [];
let animationFrameId = null; // Track animation frame for optimization

// Simplified hover state tracking
let hoveredArea = null;
const areaRadius = 100;

container.on('mousemove', (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const mouseX = event.clientX - rect.left - currentTransform.x;
    const mouseY = event.clientY - rect.top - currentTransform.y;
    
    const scaledX = mouseX / currentTransform.k;
    const scaledY = mouseY / currentTransform.k;
    
    hoveredArea = { x: scaledX, y: scaledY };
    
    // Only request new frame if not already pending
    if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(render);
    }
});

container.on('mouseleave', () => {
    hoveredArea = null;
    if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(render);
    }
});

// Optimized render function
function render() {
    animationFrameId = null; // Reset frame ID
    
    context.save();
    context.clearRect(0, 0, width, height);
    context.translate(currentTransform.x, currentTransform.y);
    context.scale(currentTransform.k, currentTransform.k);

    // Draw base connections first (very subtle)
    links.forEach(link => {
        const sourceNode = nodes[link.source-1];
        const targetNode = nodes[link.target-1];
        
        context.beginPath();
        context.strokeStyle = 'rgba(45, 45, 45, 1)';
        context.lineWidth = Math.max((link.value * 1.5) / currentTransform.k, 0.5);
        context.moveTo(sourceNode.x, sourceNode.y);
        context.lineTo(targetNode.x, targetNode.y);
        context.stroke();
    });

    // Draw highlighted connections if area is hovered
    if (hoveredArea) {
        links.forEach(link => {
            const sourceNode = nodes[link.source-1];
            const targetNode = nodes[link.target-1];
            
            // Calculate distance from hovered area to link
            const distToSource = Math.hypot(hoveredArea.x - sourceNode.x, hoveredArea.y - sourceNode.y);
            const distToTarget = Math.hypot(hoveredArea.x - targetNode.x, hoveredArea.y - targetNode.y);
            
            if (distToSource < areaRadius || distToTarget < areaRadius) {
                // Create gradient for highlighted connections
                const gradient = context.createLinearGradient(
                    sourceNode.x, sourceNode.y,
                    targetNode.x, targetNode.y
                );
                
                const sourceColor = d3.color(sourceNode.color);
                const targetColor = d3.color(targetNode.color);
                sourceColor.opacity = 0.3;
                targetColor.opacity = 0.3;
                
                gradient.addColorStop(0, sourceColor.toString());
                gradient.addColorStop(0.5, 'rgba(60, 60, 60, 0.2)');
                gradient.addColorStop(1, targetColor.toString());
                
                context.beginPath();
                context.strokeStyle = gradient;
                context.lineWidth = Math.max((link.value * 2) / currentTransform.k, 1);
                context.moveTo(sourceNode.x, sourceNode.y);
                context.lineTo(targetNode.x, targetNode.y);
                context.stroke();
            }
        });
    }

    context.restore();
}

// Update initial zoom to show the entire universe with more space
svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8));

function calculateGroupPositions(width, height, groups) {
    const positions = [];
    const centerX = width / 2;
    const centerY = height / 2;
    const universeSize = Math.min(width, height) * 1;
    
    // Define spiral parameters
    const spiralArms = 3;
    const rotationFactor = 2 * Math.PI;
    const spiralTightness = 0.3;
    
    // Generate positions for each group
    for (let i = 0; i < groups; i++) {
        const angle = (i / groups) * 2 * Math.PI;
        const galaxyRadius = universeSize * 0.3;
        
        // Add spiral effect with random perturbations
        const spiralAngle = angle + (i / groups) * rotationFactor;
        const radiusOffset = Math.random() * 0.3 + 0.7; // 70-100% of radius
        const x = centerX + Math.cos(spiralAngle) * galaxyRadius * radiusOffset;
        const y = centerY + Math.sin(spiralAngle) * galaxyRadius * radiusOffset;
        
        positions.push({ 
            x, 
            y,
            rotation: spiralAngle,
            radiusFactor: radiusOffset
        });
    }
    
    return positions;
}

fetch('/graph/')
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }

        nodes = data.nodes;
        links = data.links;

        // Calculate initial positions
        const groupPositions = calculateGroupPositions(width, height, data.groups);
        
        // Position nodes initially
        nodes.forEach(node => {
            const groupCenter = groupPositions[node.group];
            const spiralOffset = Math.random() * 2 * Math.PI;
            const distanceFromCenter = Math.pow(Math.random(), 0.5) * 150; // More nodes toward edges
            
            // Add spiral arm effect
            const spiralAngle = groupCenter.rotation + (distanceFromCenter * 0.01);
            const wobble = (Math.random() - 0.5) * 30; // Random perturbation
            
            node.x = groupCenter.x + 
                Math.cos(spiralAngle) * (distanceFromCenter + wobble) * groupCenter.radiusFactor;
            node.y = groupCenter.y + 
                Math.sin(spiralAngle) * (distanceFromCenter + wobble) * groupCenter.radiusFactor;
        });

        // Create force simulation with reduced spacing
        const simulation = d3.forceSimulation(nodes)
            .force('charge', d3.forceManyBody().strength(-50)) // Reduced repulsion
            .force('collide', d3.forceCollide().radius(d => {
                return d.size + (d.beingDragged ? 5 : 2); // Much tighter collision
            }))
            .force('x', d3.forceX(width / 2).strength(0.02))
            .force('y', d3.forceY(height / 2).strength(0.02))
            .stop();

        // Run simulation for fewer iterations
        for (let i = 0; i < 200; i++) simulation.tick();

        // Update node styling
        const nodeElements = g.append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('class', 'node')
            .attr('r', d => d.size)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .style('fill', d => {
                const color = d3.color(d.color);
                color.opacity = d.opacity;
                return color.toString();
            })
            .style('filter', 'url(#glow)');
            
        const nodeLabels = g.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('class', 'node-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .style('fill', '#ffffff')
            .style('font-size', d => d.size * 0.2) // Ajusta el tamaño del texto a algo más pequeño
            .style('display', d => d.size > 20 ? 'block' : 'none'); // Visibilidad inicial
        
        // Divide el texto en múltiples líneas con cada dos palabras y ajusta el interlineado dinámicamente
        nodeLabels.each(function(d) {
            const words = d.label.split(" "); // Divide el nombre en palabras
            const lines = [];

            // Agrupa cada dos palabras
            for (let i = 0; i < words.length; i += 2) {
                lines.push(words.slice(i, i + 2).join(" ")); // Une dos palabras
            }

            // Ajusta el tamaño de la fuente (puedes ajustar el valor según tus preferencias)
            const fontSize = d.size * 0.2; // Tamaño de fuente en píxeles
            const lineHeight = fontSize * 0.9; // Ajuste de interlineado basado en tamaño de fuente (20% adicional)

            // Añade cada línea al nodo con interlineado dinámico
            lines.forEach((line, i) => {
                d3.select(this)
                    .append("tspan")
                    .attr("x", d.x)
                    .attr("y", d.y + i * lineHeight - ((lines.length - 1) * lineHeight) / 2) // Centra verticalmente
                    .style("font-size", `${fontSize}px`) // Aplica el tamaño de fuente
                    .text(line);
            });
        });


        //
        //
        //
        // Add glow filter
        const defs = svg.append('defs');
        const filter = defs.append('filter')
            .attr('id', 'glow');
        
        filter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur');
        
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode')
            .attr('in', 'coloredBlur');
        feMerge.append('feMergeNode')
            .attr('in', 'SourceGraphic');

        // Modified drag handler with tighter collision detection
        const dragHandler = d3.drag()
            .on('start', (event, d) => {
                d.beingDragged = true;
                d3.select(event.sourceEvent.target).classed('dragging', true);
            })
            .on('drag', (event, d) => {
                const proposedX = event.x;
                const proposedY = event.y;
                let canMove = true;

                // Check collisions with group-aware spacing
                nodes.forEach(other => {
                    if (other !== d) {
                        const dx = other.x - proposedX;
                        const dy = other.y - proposedY;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        // Reduce minimum distance for same group
                        const padding = d.group === other.group ? 2 : 8;
                        const minDistance = d.size + other.size + padding;

                        if (distance < minDistance) {
                            canMove = false;
                        }
                    }
                });

                // Keep nodes within bounds with tighter margins
                const margin = d.size + 5;
                if (proposedX < margin || proposedX > width - margin ||
                    proposedY < margin || proposedY > height - margin) {
                    canMove = false;
                }

                if (canMove) {
                    d.x = proposedX;
                    d.y = proposedY;
                    d3.select(event.sourceEvent.target)
                        .attr('cx', d.x)
                        .attr('cy', d.y);
                    
                    // Update label position
                    const label = d3.selectAll('.node-label')
                        .filter(label => label === d);
                    
                    label.selectAll('tspan')
                        .attr('x', d.x)
                        .attr('y', function(_, i) {
                            const lines = label.selectAll('tspan').size();
                            const lineHeight = d.size * 0.2 * 0.9;
                            return d.y + i * lineHeight - ((lines - 1) * lineHeight) / 2;
                        });

                    requestAnimationFrame(render);
                }
            })
            .on('end', (event, d) => {
                d.beingDragged = false;
                d3.select(event.sourceEvent.target).classed('dragging', false);
            });

        // Initial render
        requestAnimationFrame(render);
    })
    .catch(error => console.error('Error:', error));

// Fix zoom and pan by adding overlay
svg.style('pointer-events', 'all')
   .call(zoom);

// Add reset zoom button functionality
document.getElementById('resetZoom').addEventListener('click', () => {
    svg.transition()
       .duration(750)
       .call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8));
});