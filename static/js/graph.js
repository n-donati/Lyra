const width = window.innerWidth;
const height = window.innerHeight - 60;

const canvas = d3.select('#graph-container')
    .append('canvas')
    .attr('width', width)
    .attr('height', height)
    .style('background', '#000');

const context = canvas.node().getContext('2d');

const svg = d3.select('#graph-container')
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

function render() {
    context.save();
    context.clearRect(0, 0, width, height);
    context.translate(currentTransform.x, currentTransform.y);
    context.scale(currentTransform.k, currentTransform.k);

    // Draw links with enhanced glass effects
    links.forEach(link => {
        const sourceNode = nodes[link.source-1];
        const targetNode = nodes[link.target-1];
        
        const gradient = context.createLinearGradient(
            sourceNode.x, sourceNode.y,
            targetNode.x, targetNode.y
        );
        
        const sourceColor = d3.color(sourceNode.color);
        const targetColor = d3.color(targetNode.color);
        sourceColor.opacity = 1; // Increased opacity
        targetColor.opacity = 1; // Increased opacity
        
        gradient.addColorStop(0, sourceColor.toString());
        gradient.addColorStop(0.1, 'rgba(64, 64, 64, 0.5)'); // Increased middle opacity
        gradient.addColorStop(0.9, 'rgba(64, 64, 64, 0.5)'); // Increased middle opacity
        gradient.addColorStop(1, targetColor.toString());

        context.beginPath();
        context.strokeStyle = gradient;
        context.lineWidth = (link.value * 3.5) / currentTransform.k; // Increased base thickness
        context.moveTo(sourceNode.x, sourceNode.y);
        context.lineTo(targetNode.x, targetNode.y);
        context.stroke();
    });

    context.restore();
}

function calculateGroupPositions(width, height, groups) {
    const positions = [];
    const centerX = width / 2;
    const centerY = height / 2;
    const universeSize = Math.min(width, height) * 1.5;
    
    // Define distinct regions for different types of clusters
    const regions = [
        // Frontal regions (top)
        { x: centerX - universeSize * 0.3, y: centerY - universeSize * 0.4, size: 0.7 },
        { x: centerX, y: centerY - universeSize * 0.35, size: 0.6 },
        { x: centerX + universeSize * 0.3, y: centerY - universeSize * 0.4, size: 0.7 },

        // Temporal regions (right)
        { x: centerX + universeSize * 0.4, y: centerY - universeSize * 0.15, size: 0.6 },
        { x: centerX + universeSize * 0.45, y: centerY, size: 0.7 },
        { x: centerX + universeSize * 0.4, y: centerY + universeSize * 0.15, size: 0.6 },

        // Parietal regions (left)
        { x: centerX - universeSize * 0.4, y: centerY - universeSize * 0.15, size: 0.6 },
        { x: centerX - universeSize * 0.45, y: centerY, size: 0.7 },
        { x: centerX - universeSize * 0.4, y: centerY + universeSize * 0.15, size: 0.6 },

        // Occipital regions (bottom)
        { x: centerX - universeSize * 0.3, y: centerY + universeSize * 0.4, size: 0.7 },
        { x: centerX, y: centerY + universeSize * 0.35, size: 0.6 },
        { x: centerX + universeSize * 0.3, y: centerY + universeSize * 0.4, size: 0.7 },

        // Motor and sensory regions (center-right)
        { x: centerX + universeSize * 0.2, y: centerY - universeSize * 0.1, size: 0.5 },
        { x: centerX + universeSize * 0.2, y: centerY + universeSize * 0.1, size: 0.5 },
        { x: centerX + universeSize * 0.15, y: centerY, size: 0.5 },

        // Deep brain regions (center-left)
        { x: centerX - universeSize * 0.2, y: centerY - universeSize * 0.1, size: 0.5 },
        { x: centerX - universeSize * 0.2, y: centerY + universeSize * 0.1, size: 0.5 },
        { x: centerX - universeSize * 0.15, y: centerY, size: 0.5 }
    ];

    // Generate positions for each group
    for (let i = 0; i < groups; i++) {
        const region = regions[i];
        const galaxyType = i % 3; // Alternate between different galaxy types
        let x = region.x, y = region.y;

        // Add specific formation patterns based on region type
        switch (galaxyType) {
            case 0: // Spiral formation
                x += (Math.random() - 0.5) * region.size * 100;
                y += (Math.random() - 0.5) * region.size * 100;
                break;
            case 1: // Elliptical formation
                const angle = Math.random() * Math.PI * 2;
                x += Math.cos(angle) * region.size * 50;
                y += Math.sin(angle) * region.size * 50;
                break;
            case 2: // Cluster formation
                x += (Math.random() - 0.5) * region.size * 150;
                y += (Math.random() - 0.5) * region.size * 150;
                break;
        }

        positions.push({ x, y });
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

        // Store data globally
        nodes = data.nodes;
        links = data.links;

        // Calculate positions
        const groupPositions = calculateGroupPositions(width, height, data.groups);
        
        // Position nodes with more realistic star cluster distribution
        nodes.forEach(node => {
            const groupCenter = groupPositions[node.group];
            const angle = Math.random() * Math.PI * 2;
            
            // Create dense star clusters with varying sizes
            const clusterDensity = Math.random();
            const clusterSize = 50 + (node.group * 15) + (clusterDensity * 100);
            const distributionRadius = Math.pow(Math.random(), 2) * clusterSize; // Quadratic distribution for denser cores
            
            node.x = groupCenter.x + Math.cos(angle) * distributionRadius;
            node.y = groupCenter.y + Math.sin(angle) * distributionRadius;
        });

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
                color.opacity = 0.7;
                return color.toString();
            })
            .style('filter', 'url(#glow)');
            
        const nodeLabels = g.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('class', 'node-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y - d.size - 5) // Ajuste para que el texto esté encima del círculo
            .attr('text-anchor', 'middle')
            .style('fill', '#ffffff')
            .style('display', d => d.size > 20 ? 'block' : 'none') // Visibilidad inicial
            .text(d => "TEXT");


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

        // Handle node dragging with position updates
        const dragHandler = d3.drag()
            .on('start', (event, d) => {
                d3.select(event.sourceEvent.target).classed('dragging', true);
            })
            .on('drag', (event, d) => {
                d.x = event.x;
                d.y = event.y;
                d3.select(event.sourceEvent.target)
                    .attr('cx', d.x)
                    .attr('cy', d.y);
                requestAnimationFrame(render);
            })
            
            .on('end', (event, d) => {
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

// Update initial zoom to show the entire universe with more space
svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8));