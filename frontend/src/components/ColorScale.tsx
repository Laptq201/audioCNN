const ColorScale = ({
    width = 200, height = 16, min = -1, max = 1
}: {
    width?: number;
    height?: number; 
    min?: number; 
    max?: number
}) => {
    const gradient = "linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)";

    return <div className="flex items-center gap-3"></div>;
}


export default ColorScale;
